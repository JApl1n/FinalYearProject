import hoomd
import hoomd.md
import numpy as np
from mpi4py import MPI

class PrintTimestep(hoomd.custom.Action):
    def act(self, timestep):
        print(timestep)


class RodPropulsion(hoomd.custom.Action):
    def __init__(self, numRods, rodLength, numSolvent, boxSizes, forceMagnitude, dt, communicator):
        self._numRods = numRods
        self._rodLength = rodLength
        self._numSolvent = numSolvent
        self._boxSizes = boxSizes
        self._forceMagnitude = forceMagnitude
        self._dt = dt
        self._communicator = communicator
        self._rank = communicator.rank
        self._size = communicator.num_ranks

        # Assign rods to MPI ranks
        self._rodsPerRank = numRods // self._size
        self._extraRods = numRods % self._size
        self._rodStart = (self._rank * self._rodsPerRank) + min(self._rank, self._extraRods)
        self._rodEnd = self._rodStart + self._rodsPerRank + (1 if self._rank < self._extraRods else 0)

        self._mpiComm = MPI.COMM_WORLD

    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            localData = {
                "positions": snap.particles.position,
                "tags": snap.particles.tag,
                "velocities": snap.particles.velocity}

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
      
                rodIndices = [i for i, tag in enumerate(globalTags) if rodStartTag <= tag < rodEndTag]
                
                if not rodIndices:
                    continue

                rodPositions = globalPositions[rodIndices]
                rodAxis = rodPositions[-1] - rodPositions[0]
                
                for i in range(3):
                    if rodAxis[i] > self._boxSizes[i] / 2:
                        rodAxis[i] -= self._boxSizes[i]
                    elif rodAxis[i] < -self._boxSizes[i] / 2:
                        rodAxis[i] += self._boxSizes[i]
                
                rodAxis /= np.linalg.norm(rodAxis)

                # Apply propulsion force along the rod's axis
                for i, index in enumerate(rodIndices):
                    velocityIncrement = self._forceMagnitude * rodAxis * self._dt
                    globalVelocities[index] += velocityIncrement

            # Update local velocities
            snap.particles.velocity[:] = globalVelocities[:len(snap.particles.velocity)]


# Parameters
Lx, Ly, Lz = 20, 20, 20  # Box dimensions
dt = 0.005
drivingForceMagnitude = 10
numSolvent = 0
numRods = 3
rodLength = 5
rodSpacing = 1
simLength = 200
outStep = 25
k1 = 100
k2 = 100

# Initialize the simulation
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=42)
sim.create_state_from_gsd(filename="rodsInitial.gsd")

harmonicBond = hoomd.md.bond.Harmonic()
harmonicBond.params['rodBond'] = dict(k=k1, r0=rodSpacing)

harmonicAngle = hoomd.md.angle.Harmonic()
harmonicAngle.params['rodAngle'] = dict(k=k2, t0=np.pi)

# Define interactions
cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[("solvent", "solvent")] = dict(epsilon=1.0, sigma=1.0)
lj.params[("solvent", "rod")] = dict(epsilon=1.0, sigma=1.0)
lj.params[("rod", "rod")] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[("solvent", "solvent")] = 1.122
lj.r_cut[("solvent", "rod")] = 2.5
lj.r_cut[("rod", "rod")] = 2.5

integrator = hoomd.md.Integrator(dt=dt)
langevin = hoomd.md.methods.Langevin(kT=1.0, filter=hoomd.filter.All())
langevin.gamma['solvent'] = 1.0
langevin.gamma['rod'] = 0.5
integrator.methods.append(langevin)

integrator.forces.append(lj)
integrator.forces.append(harmonicBond)
integrator.forces.append(harmonicAngle)
sim.operations.integrator = integrator

printTimestepOperation = hoomd.write.CustomWriter(action=PrintTimestep(), trigger=hoomd.trigger.Periodic(1))
#sim.operations.writer.append(printTimestepOperation)

# Add custom updater for propulsion
forceAction = RodPropulsion(numRods, rodLength, numSolvent, [Lx, Ly, Lz], drivingForceMagnitude, dt, sim.device.communicator)
forceOperation = hoomd.update.CustomUpdater(action=forceAction, trigger=hoomd.trigger.Periodic(1))
sim.operations.updaters.append(forceOperation)

# Add GSD writer
gsdWriter = hoomd.write.GSD(
    filename="simulation_output.gsd",
    trigger=hoomd.trigger.Periodic(outStep),
    mode="wb",
)
sim.operations.writers.append(gsdWriter)

# Run simulation
sim.run(simLength)
print("Simulation complete. Output saved to 'simulation_output.gsd'.")
