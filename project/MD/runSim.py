import hoomd
import hoomd.md
import numpy as np
from mpi4py import MPI
import gsd.hoomd
import h5py
import os
import json

# This will print the timestep within the simulation after each step in the simulation.
class PrintTimestep(hoomd.custom.Action):
    def act(self, timestep):
        print(timestep)

# This class allows me to edit and read the positions of particles in the simulation and works with mpi
class RodPropulsion(hoomd.custom.Action):
    def __init__(self, numRods, rodLength, numSolvent, boxSizes, forceMagnitude, dt, outstep, outputFilename, communicator):
        self._numRods = numRods
        self._rodLength = rodLength
        self._numSolvent = numSolvent
        self._boxSizes = boxSizes
        self._forceMagnitude = forceMagnitude
        self._dt = dt
        self._outStep = outStep
        self._communicator = communicator
        self._rank = communicator.rank
        self._size = communicator.num_ranks
        self._mpiComm = MPI.COMM_WORLD

        # Assign rods to MPI ranks
        self._rodsPerRank = numRods // self._size
        self._extraRods = numRods % self._size
        self._rodStart = (self._rank * self._rodsPerRank) + min(self._rank, self._extraRods)
        self._rodEnd = self._rodStart + self._rodsPerRank + (1 if self._rank < self._extraRods else 0)


    def attatch(self, simulation):
        self._state = simulation.state

    # This function writes the types to the start fo the output of positions. This means we can identify which are solvent and which are rods
    def InitHdf5(self, filename, snap, timestep):
        
        types = "types"

        localData = {"tags": snap.particles.tag.tolist(), types: snap.particles.typeid.tolist()}
        allData = self._mpiComm.allgather(localData)
        globalTags = []
        globalTypes = []

        for data in allData:
            globalTags.extend(data["tags"])
            globalTypes.extend(data[types])
        allTags = np.array(globalTags)
        allTypes = np.array(globalTypes)

        sortedIndices = np.argsort(allTags)
        sortedTypes = allTypes[sortedIndices]

        if self._rank == 0:
            with h5py.File(filename, "a") as f:
                if types in f:
                    del f[types]
                f.create_dataset(types, data=sortedTypes)

    # This is done to update the force applied to each rod. We cant access forces in this version of hoomd (5.0) so we add velocity instead.
    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            
            localData = {
                "positions": snap.particles.position.tolist(),
                "tags": snap.particles.tag.tolist(),
                "velocities": snap.particles.velocity.tolist()}

            allData = self._mpiComm.allgather(localData)

            globalPositions = []
            globalTags = []
            globalVelocities = []

            for data in allData:
                globalPositions.extend(data["positions"])
                globalTags.extend(data["tags"])
                globalVelocities.extend(data["velocities"])

            globalPositions = np.array(globalPositions)
            globalTags = np.array(globalTags)
            globalVelocities = np.array(globalVelocities)

            for rodNum in range(self._rodStart, self._rodEnd):
                rodStartTag = self._numSolvent + rodNum * self._rodLength
                rodEndTag = rodStartTag + self._rodLength
      
                rodMask = (globalTags >= rodStartTag) & (globalTags < rodEndTag)
                rodIndices = np.where(rodMask)[0]
                                    
                if len(rodIndices) == 0:
                    continue  # Skip if no rod particles found

                rodTags = globalTags[rodIndices]
                sortedOrder = np.argsort(rodTags)
                rodIndices = rodIndices[sortedOrder]
                rodTags = rodTags[sortedOrder]

                rodPositions = globalPositions[rodIndices]
                rodAxis = rodPositions[-1] - rodPositions[0]
                                    

                for i in range(3):
                    if rodAxis[i] > self._boxSizes[i] / 2:
                        rodAxis[i] -= self._boxSizes[i]
                    elif rodAxis[i] < -self._boxSizes[i] / 2:
                        rodAxis[i] += self._boxSizes[i]

                rodAxis /= np.linalg.norm(rodAxis)

                # Apply propulsion force along the rod's axis
                velocityIncrement = self._forceMagnitude * rodAxis*self._dt
                for i, index in enumerate(rodIndices):
                    globalVelocities[index] += velocityIncrement

            # Update local velocities
            if len(globalVelocities) >= len(snap.particles.velocity):
                snap.particles.velocity[:] = globalVelocities[:len(snap.particles.velocity)]
            
            if timestep == 100:
                self.InitHdf5(outputFilename, snap, timestep)

            if self._rank == 0:
                if timestep % self._outStep == 0:
                    sortedIndices = np.argsort(globalTags)
                    sortedPositions = globalPositions[sortedIndices]
                    PositionLogger.SaveToHdf5(outputFilename, sortedPositions, timestep)


# Writes the position of all particles when called by the above function.
class PositionLogger:
    @staticmethod
    def SaveToHdf5(filename, positions, timestep):
        if positions is None:
            return

        with h5py.File(filename, "a") as f:
            datasetName = f"step_{timestep}"
            # Remove dataset if it already exists (to prevent errors)
            if datasetName in f:
                del f[datasetName]

            f.create_dataset(datasetName, data=positions)

# Edit the LJ force parameters to increase gradually over warmup period
class LJParameterTuner(hoomd.custom.Action):
    def __init__(self, ljPotential, startEpsilon, endEpsilon, totalSteps):
        self.ljPotential = ljPotential
        self.startEpsilon = startEpsilon
        self.endEpsilon = endEpsilon
        self.totalSteps = totalSteps
        self.step = 0

    def act(self, timestep):
        # Compute new epsilon using a linear ramp
        fraction = min(self.step / self.totalSteps, 1.0)
        newEpsilon = self.startEpsilon + fraction * (self.endEpsilon - self.startEpsilon)

        # Update LJ parameters
        self.ljPotential.params[("solvent", "solvent")]['epsilon'] = newEpsilon
        self.ljPotential.params[("solvent", "rod")]['epsilon'] = newEpsilon
        self.ljPotential.params[("rod", "rod")]['epsilon'] = newEpsilon

        # Increase step count
        self.step += 1

# Load data also used by initialiser
with open("simulationMetadata.json", "r") as f:
    params = json.load(f)

Lx, Ly, Lz = params["Lx"], params["Ly"], params["Lz"]
numSolvent = params["numSolvent"]
numRods = params["numRods"]
rodLength = params["rodLength"]
rodSpacing = params["rodSpacing"]

# Parameters to edit here
dt = 0.005  # Time step
drivingForceMagnitude = 10  # Magnitude of force driving rods forward
warmupLength = 100  # Number of timesteps to tune forces to prevent extreme initial velocities
simLength = 100  # Number of timesteps for run of simulation
outStep = 10  # Periodicity of output frames
kBond = 200  # Strength of force between particles in rod
kAngle = 200  # Strength to keep particles in rod aligned

outputFilename = "positions.h5"
inputFilename = "rodsInitial.gsd"

# Remove old output file
if (MPI.COMM_WORLD.rank == 0) and (os.path.isfile(outputFilename)):
    os.remove(outputFilename)

# Initialize the simulation
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=42)
sim.create_state_from_gsd(filename=inputFilename)

# Add bonds for keeping rods in a line
harmonicBond = hoomd.md.bond.Harmonic()
harmonicBond.params['rodBond'] = dict(k=kBond, r0=rodSpacing)

harmonicAngle = hoomd.md.angle.Harmonic()
harmonicAngle.params['rodAngle'] = dict(k=kAngle, t0=np.pi)

# Define interactions
cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[("solvent", "solvent")] = dict(epsilon=0.1, sigma=0.5)
lj.params[("solvent", "rod")] = dict(epsilon=0.5, sigma=0.5)
lj.params[("rod", "rod")] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[("solvent", "solvent")] = 1.122
lj.r_cut[("solvent", "rod")] = 2.5
lj.r_cut[("rod", "rod")] = 2.5

# Add some energy to the system
integrator = hoomd.md.Integrator(dt=dt)
langevin = hoomd.md.methods.Langevin(kT=1.0, filter=hoomd.filter.All())
langevin.gamma['solvent'] = 1.0
langevin.gamma['rod'] = 0.5
integrator.methods.append(langevin)

integrator.forces.append(lj)
integrator.forces.append(harmonicBond)
integrator.forces.append(harmonicAngle)
sim.operations.integrator = integrator

# Print the timestep of a simulation after that step, periodically
printTimestepOperation = hoomd.write.CustomWriter(action=PrintTimestep(), trigger=hoomd.trigger.Periodic(1))
#sim.operations.writer.append(printTimestepOperation)

# Add thermodynamic properties for tracking desnity in warm up
thermodynamicProperties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamicProperties)

# Start with smaller forces and ramp to desired values
startEpsilon = 0.001
endEpsilon = 0.5
ljTuner = LJParameterTuner(lj, startEpsilon, endEpsilon, totalSteps=warmupLength)
ljTuneOperation = hoomd.update.CustomUpdater(action=ljTuner, trigger=hoomd.trigger.Periodic(1)) 
sim.operations.updaters.append(ljTuneOperation)

# Run warm up
sim.run(warmupLength)

sim.operations.updaters.remove(ljTuneOperation)

# Add custom updater for propulsion
forceAction = RodPropulsion(numRods, rodLength, numSolvent, [Lx, Ly, Lz], drivingForceMagnitude, dt, outStep, outputFilename, sim.device.communicator)
forceOperation = hoomd.update.CustomUpdater(action=forceAction, trigger=hoomd.trigger.Periodic(1))
sim.operations.updaters.append(forceOperation)

if MPI.COMM_WORLD.rank == 0:
    print("Equilibration complete. Running main simulation...")

# Add thermodynamic properties for logging then write to h5 file
logger = hoomd.logging.Logger(categories=["scalar", "sequence"])
logger.add(thermodynamicProperties)
logger.add(sim, quantities=["timestep", "walltime"])

hdf5Writer = hoomd.write.HDF5Log(
    trigger=hoomd.trigger.Periodic(outStep), filename="log.h5", mode="w", logger=logger)
sim.operations.writers.append(hdf5Writer)


# Run simulation
sim.run(simLength)


if MPI.COMM_WORLD.rank == 0:
    print(f"Simulation complete. Output saved to {outputFilename}.")
    print(f'Performance: {sim.tps} timesteps per second')
