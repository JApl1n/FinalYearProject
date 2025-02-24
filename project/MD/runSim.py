import hoomd
import hoomd.md
import numpy as np
from mpi4py import MPI
import gsd.hoomd
import h5py
import os
import json
import sys

# This will print the timestep within the simulation after each step in the simulation.
class PrintTimestep(hoomd.custom.Action):
    def act(self, timestep):
        if MPI.COMM_WORLD.rank == 0:
            print(timestep)

# This class allows me to edit and read the positions of particles in the simulation and works with mpi
class RodPropulsion(hoomd.custom.Action):
    def __init__(self, numRods, rodLength, numSolvents, boxSizes, forceMagnitude, dt, outstep, outputFilename, warmupLength, communicator):
        self._numRods = numRods
        self._rodLength = rodLength
        self._numSolvents = numSolvents
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
        
        datasetName = "metadata"

        localData = {"tags": snap.particles.tag.tolist(), "types": snap.particles.typeid.tolist()}
        allData = self._mpiComm.allgather(localData)
        globalTags = []
        globalTypes = []

        for data in allData:
            globalTags.extend(data["tags"])
            globalTypes.extend(data["types"])
        allTags = np.array(globalTags)
        allTypes = np.array(globalTypes)

        sortedIndices = np.argsort(allTags)
        sortedTypes = allTypes[sortedIndices]

        if self._rank == 0:
            with h5py.File(filename, "a") as f:
                if datasetName in f:
                    del f[datasetName]
                f.create_dataset(datasetName, data=sortedTypes)
                
    # This is done to update the force applied to each rod. We cant access forces in this version of hoomd (5.0) so we add velocity instead.
    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            
            localPositions = snap.particles.position[:]
            localTags = snap.particles.tag[:]
            localVelocities = snap.particles.velocity[:]

            allData = self._mpiComm.allgather({
                "positions": localPositions,
                "tags": localTags,
                "velocities": localVelocities})

        globalPositions = np.concatenate([data["positions"] for data in allData])
        globalTags = np.concatenate([data["tags"] for data in allData])
        globalVelocities = np.concatenate([data["velocities"] for data in allData])
            
        sortedIndices = np.argsort(globalTags)
        sortedGlobalTags = globalTags[sortedIndices]
        sortedGlobalPositions = globalPositions[sortedIndices]
        sortedGlobalVelocities = globalVelocities[sortedIndices]

        velocityUpdates = np.zeros_like(sortedGlobalVelocities)
        
        for rodNum in range(self._numRods):
            rodStartTag = self._numSolvents + rodNum * self._rodLength
            rodEndTag = rodStartTag + self._rodLength

            rodMask = (sortedGlobalTags >= rodStartTag) & (sortedGlobalTags < rodEndTag)
            rodIndices = np.where(rodMask)[0]
        
            if len(rodIndices) == 0:
                continue  # No particles for this rod
            # (They should already be in sorted order, but we can enforce it)
            rodTags = sortedGlobalTags[rodIndices]
            
            rodPositions = sortedGlobalPositions[rodIndices]

            # Compute the rod axis as the difference between the last and first particle.
            rodAxis = rodPositions[-1] - rodPositions[0]    

            # Apply periodic boundary corrections (assuming self._boxSizes is [Lx, Ly, Lz])
            for i in range(3):
                if rodAxis[i] > self._boxSizes[i] / 2:
                    rodAxis[i] -= self._boxSizes[i]
                elif rodAxis[i] < -self._boxSizes[i] / 2:
                    rodAxis[i] += self._boxSizes[i]
            # Normalize the axis
            rodAxis /= np.linalg.norm(rodAxis)

            velocityIncrement = self._forceMagnitude * rodAxis * dt

            # Add this increment to each particle in the rod.
            for idx in rodIndices:
                velocityUpdates[idx] += velocityIncrement

        
        # Now, send the computed updates to the ranks that own each particle.
        # Build a mapping of tags to owner rank.
        tagToRank = {}
        for rank, data in enumerate(allData):
            for tag in data["tags"]:
                tagToRank[tag] = rank  # This mapping may change as domains evolve.

        # Prepare the sendData dictionary.
        sendData = {rank: [] for rank in range(self._mpiComm.size)}
        for i, tag in enumerate(sortedGlobalTags):
            ownerRank = tagToRank.get(tag, None)
            # Only send the update if the owner is different from the current rank.
            if ownerRank is not None and ownerRank != self._mpiComm.rank:
                sendData[ownerRank].append((tag, velocityUpdates[i]))

        
        # Exchange updates with other ranks.
        # Use sendrecv for each rank that we have updates to exchange.
        recvData = {rank: [] for rank in range(self._mpiComm.size)}
        for rank in sendData.keys():
            if rank == self._mpiComm.rank:
                continue
            sendBuffer = sendData[rank]
            # Exchange updates with rank.
            recvBuffer = self._mpiComm.sendrecv(sendobj=sendBuffer, dest=rank, source=rank)
            recvData[rank] = recvBuffer


        # Apply received velocity updates to local particles.
        with self._state.cpu_local_snapshot as snap:
            localTags = snap.particles.tag[:]  # Local tags on this rank.
            for rank in recvData:
                for tag, velUpdate in recvData[rank]:
                    # Only apply if this rank owns the particle.
                    if tag in localTags:
                        localIndex = np.where(localTags == tag)[0][0]
                        snap.particles.velocity[localIndex] += velUpdate
                        
            
        if timestep == (warmupLength+1):
            with self._state.cpu_local_snapshot as snap:
                self.InitHdf5(outputFilename, snap, timestep)

        if self._rank == 0:
            if timestep % self._outStep == 0:
                PositionLogger.SaveToHdf5(outputFilename, sortedGlobalPositions, timestep)

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
    def __init__(self, ljPotential, startEpsilon, endEpsilon, totalSteps, alpha=0.5):
        self.ljPotential = ljPotential
        self.startEpsilon = startEpsilon
        self.endEpsilon = endEpsilon
        self.totalSteps = totalSteps
        self.alpha = alpha
        self.step = 0

    def act(self, timestep):
        # Compute new epsilon using an exponential or linear ramp
        fraction = 1 - np.exp(-self.alpha*self.step/self.totalSteps)
        #fraction = min(self.step+1/self.totalSteps, 1.0)

        for pair in self.ljPotential.params.keys():
            startEpsilon = self.startEpsilon[pair]
            endEpsilon = self.endEpsilon[pair]
            newEpsilon = startEpsilon + fraction * (endEpsilon - startEpsilon)
            self.ljPotential.params[pair]["epsilon"] = newEpsilon
        
        # Increase step count
        self.step += 1


class VelocityResetter(hoomd.custom.Action):
    def act(self, timestep):
        with sim.state.cpu_local_snapshot as snap:
            snap.particles.velocity[:] = np.zeros_like(snap.particles.velocity)



inputs = sys.argv
ID = ""  # Default to blank

if (len(inputs) > 1):
    inp = inputs[1]  # Assume ID given by first argument, ignore rest
    if ("=" in inp):
        param, value = inp.split("=")[0], inp.split("=")[1]
        if (param == "ID"):
            ID = value



# Load data also used by initialiser
with open(f"simulationMetadata{ID}.json", "r") as f:
    params = json.load(f)

# Parameters from initialisation
Lx, Ly, Lz = params["Lx"], params["Ly"], params["Lz"]
numSolvents = params["numSolvents"]
numRods = params["numRods"]
rodLength = params["rodLength"]
rodSpacing = params["rodSpacing"]

# Parameters editable here, but inherited initially from init.py
dt = params["dt"]  # Time step
dtWarmup = params["dtWarmup"]
drivingForceMagnitude = params["dtWarmup"] # Magnitude of force driving rods forward
warmupLength = params["warmupLength"]  # Number of timesteps to tune forces to prevent extreme initial velocities
simLength = params["simLength"]  # Number of timesteps for run of simulation
outStep = params["outStep"]  # Periodicity of output frames
kBond = params["kBond"]  # Strength of force between particles in rod
kAngle = params["kAngle"]  # Strength of force keeping particles in rod aligned
sigma = params["sigma"]  # Distance over which the leonard jones potential spreads
kT = params["kT"]  # Initial kinetic energy given to system after warmup
gammaSolvent = params["gammaSolvent"]  # Slight resistance given to solvent particles
gammaRod = params["gammaRod"]  # Slight resistance given to rod particles

ssei = params["ssei"]
rsei = params["rsei"]
rrei = params["rrei"]
ssef = params["ssef"]
rsef = params["rsef"]
rref = params["rref"]

ID = params["ID"]  # Might as well assign again

startEpsilons = {
    ("solvent", "solvent"): ssei,
    ("rod", "solvent"): rsei,
    ("rod", "rod"): rrei}
endEpsilons = {
    ("solvent", "solvent"): ssef,
    ("rod", "solvent"): rsef,
    ("rod", "rod"): rref}




outputFilename = f"positions{ID}.h5"
inputFilename = f"rodsInitial{ID}.gsd"

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


### WARMUP SEQUENCE

# Define interaction forces at start with desired ending values.
# The start values are small for warmup to drift particles from each other,
# then end at larger values for desired inter-particle forces.
startEpsilons = {
    ("solvent", "solvent"): 0.00001,
    ("rod", "solvent"): 0.00005,
    ("rod", "rod"): 0.0001}

endEpsilons = {
    ("solvent", "solvent"): 0.5,
    ("rod", "solvent"): 0.5,
    ("rod", "rod"): 1}


cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
for pair, epsilon in startEpsilons.items():
    lj.params[pair] = dict(epsilon=epsilon, sigma=1.0)
lj.r_cut[("solvent", "solvent")] = 1.122
lj.r_cut[("rod", "solvent")] = 2.5
lj.r_cut[("rod", "rod")] = 2.5

# Start with smaller forces and ramp to desired values
ljTuner = LJParameterTuner(lj, startEpsilons, endEpsilons, totalSteps=warmupLength)
ljTuneOperation = hoomd.update.CustomUpdater(action=ljTuner, trigger=hoomd.trigger.Periodic(1)) 
sim.operations.updaters.append(ljTuneOperation)


# Add forces to integrator for system
integrator = hoomd.md.Integrator(dt=dtWarmup)
integrator.forces.append(lj)
integrator.forces.append(harmonicBond)
integrator.forces.append(harmonicAngle)
sim.operations.integrator = integrator

# Add displacement cap
disCap = hoomd.md.methods.DisplacementCapped(
        filter=hoomd.filter.All(),
        maximum_displacement=0.00000001)
integrator.methods.append(disCap)

# Reset velocities periodically, we couldnt see a velocity cap so this will do.
resetterAction = VelocityResetter()
velResetter = hoomd.update.CustomUpdater(action=resetterAction, trigger=hoomd.trigger.Periodic(10))
sim.operations.updaters.append(velResetter)

# Print the timestep of a simulation after that step, periodically
printTimestepOperation = hoomd.write.CustomWriter(action=PrintTimestep(), trigger=hoomd.trigger.Periodic(outStep))
sim.operations.writers.append(printTimestepOperation)

# Add thermodynamic properties for tracking
thermodynamicProperties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamicProperties)


## Run warm up with increasing lj and minimal displacements
sim.run(warmupLength)



### Now forces at full values, remove warming functions and parameters to prepare to run full simulation
integrator.methods.remove(disCap)
sim.operations.updaters.remove(ljTuneOperation)
sim.operations.updaters.remove(velResetter)

with sim.state.cpu_local_snapshot as snap:
    snap.particles.velocity[:] = np.zeros_like(snap.particles.velocity)




integrator.dt = dt
langevinNormal = hoomd.md.methods.Langevin(kT=kT, filter=hoomd.filter.All())
langevinNormal.gamma['solvent'] = gammaSolvent
langevinNormal.gamma['rod'] = gammaRod
integrator.methods.append(langevinNormal)

# Add custom updater for propulsion
forceAction = RodPropulsion(numRods, rodLength, numSolvents, [Lx, Ly, Lz], drivingForceMagnitude, dt, outStep, outputFilename, warmupLength, sim.device.communicator)
forceOperation = hoomd.update.CustomUpdater(action=forceAction, trigger=hoomd.trigger.Periodic(1))
sim.operations.updaters.append(forceOperation)


# Add thermodynamic properties for logging then write to h5 file
logger = hoomd.logging.Logger(categories=["scalar", "sequence"])
logger.add(thermodynamicProperties)
logger.add(sim, quantities=["timestep", "walltime"])

hdf5Writer = hoomd.write.HDF5Log(
    trigger=hoomd.trigger.Periodic(outStep), filename=f"log{ID}.h5", mode="w", logger=logger)
sim.operations.writers.append(hdf5Writer)


# Run simulation
if MPI.COMM_WORLD.rank == 0:
    print("Equilibration complete. Running main simulation...")

sim.run(simLength)


if MPI.COMM_WORLD.rank == 0:
    print(f"Simulation complete. Output saved to {outputFilename}.")
    print(f'Performance: {sim.tps} timesteps per second')
