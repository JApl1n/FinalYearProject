import hoomd
import hoomd.md
import numpy as np



class PrintTimestep(hoomd.custom.Action):
    def act(self, timestep):
        print(timestep)


class RodPropulsion(hoomd.custom.Action):
    def __init__(self, numRods, rodLength, numSolvent, boxSizes, forceMagnitude, dt):
        self._numRods = numRods
        self._rodLength = rodLength
        self._numSolvent = numSolvent
        self._boxSizes = boxSizes
        self._forceMagnitude = forceMagnitude
        self._dt = dt 


    def attach(self, simulation):
        self._state = simulation.state
        self._comm = simulation.device.communicator

    def detach(self):
        del self._state
        del self._comm


    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            for rodNum in range(self._numRods):
                # Get the starting tag index for this rod
                rodStartTag = self._numSolvent + rodNum * self._rodLength
                rodEndTag = rodStartTag + self._rodLength

                # Retrieve positions and velocities for this rod using the tag-to-index map
                rodIndices = [snap.particles.rtag[tag] for tag in range(rodStartTag, rodEndTag) if snap.particles.rtag[tag] < snap.particles.position.shape[0]]

                # Skip if this rank owns none of the rod's particles
                if not rodIndices:
                    continue

                rodPositions = snap.particles.position[rodIndices]
                #print(f"Rank {self._comm.rank}: Rod {rodNum} local positions: {rodPositions}")
            
                # Compute the rod's axis
                rodAxis = rodPositions[-1] - rodPositions[0]
                # Adjust direction for boundary crossing in each dimension
                for i in range(3):
                    # If the distance between particles exceeds half the box size, apply PBC
                    if rodAxis[i] > self._boxSizes[i]/2:
                        rodAxis[i] -= self._boxSizes[i]  # Wrap around
                    elif rodAxis[i] < -self._boxSizes[i]/2:
                        rodAxis[i] += self._boxSizes[i]
                
                rodAxis /= np.linalg.norm(rodAxis)  # Normalize to unit vector

                # Apply forces along the rod axis
                for i, globalIndex in enumerate(rodIndices):
                    velocityIncrement = self._forceMagnitude * rodAxis * self._dt
                    snap.particles.velocity[globalIndex] += velocityIncrement


def GenerateRandomRods(numRods, rodLength, boxSize, rodSpacing):
    rods=[]
    for _ in range(numRods):
        while True:
            # Random starting position in box
            low = -boxSize/2
            high = boxSize/2
            startPos = np.random.uniform(low=low, high=high, size=3)

            # Random orientation
            orientation = np.random.uniform(-1,1,size=3)
            orientation /= np.linalg.norm(orientation)

            # Generate position of particles along rod
            rodPositions = [startPos + i * rodSpacing * orientation for i in range(rodLength)]
            rodPositions = np.array(rodPositions)

            # Check if all positions are inside the box
            if np.all((rodPositions >= low) & (rodPositions <= high)):
                rods.extend(rodPositions)
                break

    return np.array(rods)


# Tunable parameters
Lx, Ly, Lz = 20, 20, 20  # Size of box dimensions
dt = 0.005  # Time step
drivingForceMagnitude = 1.0  # Size of force drving rods forward
numSolvent = 100  # Number of solvent particles
numRods = 5  # Number of whole rods
rodLength = 4  # Number of particles per rod
rodSpacing = 1.0  # Distance between rod particles
k1 = 100.0  # Stiffness of separation springs between particles
k2 = 100.0  # Stiffness of alignment springs between particles
buff = 0.4  # Separation of particles from each other to check
outStep = 15  # Output to writer every this many steps
simLength = 300  # Number of timesteps to run

# Initialize HOOMD simulation
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=42)

box = hoomd.Box(Lx=Lx, Ly=Ly, Lz=Lz)

# Total number of particles
numParticles= numSolvent + numRods * rodLength

# Create snapshot
snapshot = hoomd.Snapshot()
snapshot.particles.N = numParticles

# Uniformly position solvent particles
solventPositions = np.random.uniform(low=-Lx/2, high=Lx/2, size=(numSolvent, 3))


## Define rod positions (align along the z-axis for simplicity)
#rod1Positions = [[0, 0, z] for z in np.linspace(-2, 2, rodLength)]
#rod2Positions = [[5, 5, z] for z in np.linspace(-2, 2, rodLength)]  # Second rod shifted in space
## Combine with solvent positions
#snapshot.particles.position[:] = np.vstack([solventPositions, rod1Positions, rod2Positions])


# Generate random rods
rodPositions = GenerateRandomRods(numRods, rodLength, Lx, rodSpacing)
snapshot.particles.position[:] = np.vstack([solventPositions, rodPositions])


snapshot.particles.types = ['solvent', 'rod']

# Define harmonic bond potential
harmonicBond = hoomd.md.bond.Harmonic()
harmonicBond.params['rodBond'] = dict(k=k1, r0=rodSpacing)  # Stiff spring, rest length = rod_spacing

# Set up bonds in the snapshot
snapshot.bonds.N = numRods * (rodLength - 1)  # Bonds for all rods
snapshot.bonds.types = ['rodBond']

# Define the bonds (connect consecutive particles in each rod)
bondList = []
for i in range(numRods):
    startIndex = numSolvent + i * rodLength  # Starting index of the current rod
    for j in range(rodLength - 1):
        bondList.append([startIndex + j, startIndex + j + 1])

snapshot.bonds.group[:] = bondList
snapshot.bonds.typeid[:] = [0] * len(bondList)

# Update particle types (assign all rod particles the same type)
snapshot.particles.typeid[:] = [0] * numSolvent + [1] * (rodLength * numRods)



# Define harmonic angular potential
harmonicAngle = hoomd.md.angle.Harmonic()
harmonicAngle.params['rodAngle'] = dict(k=k2, t0=np.pi)  # Example values: k=stiffness, t0=equilibrium angle

# Set up angles in the snapshot
numAngles = numRods * (rodLength - 2)  # Number of angles per rod
snapshot.angles.N = numAngles
snapshot.angles.types = ['rodAngle']

# Define angles between consecutive triplets of particles in each rod
angleList = []
for i in range(numRods):
    startIndex = numSolvent + i * rodLength
    for j in range(rodLength - 2):
        angleList.append([startIndex + j, startIndex + j + 1, startIndex + j + 2])

snapshot.angles.group[:] = angleList
snapshot.angles.typeid[:] = [0] * len(angleList)




# Set the simulation box
snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]

# Initialize simulation state from the snapshot
sim.create_state_from_snapshot(snapshot)

# Define neighbor list
cell = hoomd.md.nlist.Cell(buffer=buff)

# Define LJ interactions
lj = hoomd.md.pair.LJ(nlist=cell)

# Define solvent-solvent interactions (WCA potential)
lj.params[("solvent", "solvent")] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[("solvent", "solvent")] = 1.122  # WCA cutoff

# Define solvent-rod interactions (LJ potential)
lj.params[("solvent", "rod")] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[("solvent", "rod")] = 2.5

# Rod-rod interaction (optional, can be zero or specified explicitly)
lj.params[("rod", "rod")] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[("rod", "rod")] = 2.5

# Langevin thermostat for solvent and rods
integrator = hoomd.md.Integrator(dt=dt)
langevin = hoomd.md.methods.Langevin(kT=1.0, filter=hoomd.filter.All())
langevin.gamma['solvent'] = 1.0
langevin.gamma['rod'] = 0.5  # Adjust drag for larger particles
integrator.methods.append(langevin)

# Add the interaction forces to the integrator
integrator.forces.append(lj)
integrator.forces.append(harmonicBond)
integrator.forces.append(harmonicAngle)
sim.operations.integrator = integrator

# Add timestep for progress while simulating
printTimestepOperation = hoomd.write.CustomWriter(action=PrintTimestep(), trigger=hoomd.trigger.Periodic(1))
#sim.operations.writers.append(printTimestepOperation)


drivingForceMagnitude = 10
forceAction = RodPropulsion(numRods, rodLength, numSolvent, [Lx,Ly,Lz], drivingForceMagnitude, dt)
forceOperation = hoomd.update.CustomUpdater(action=forceAction, trigger=hoomd.trigger.Periodic(1))
sim.operations.updaters.append(forceOperation)


# Add a GSD writer for output
gsdWriter = hoomd.write.GSD(
    filename="simulation_output.gsd",
    trigger=hoomd.trigger.Periodic(outStep),  # Save every out steps
    mode="wb",
)
sim.operations.writers.append(gsdWriter)

# Run the simulation
sim.run(simLength)

print("Simulation complete. GSD file saved as 'simulation_output.gsd'.")
