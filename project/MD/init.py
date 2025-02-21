import hoomd
import numpy as np
import json
import sys

def GenerateRandomRods(numRods, rodLength, boxSize, rodSpacing):
    rods = []
    for _ in range(numRods):
        while True:
            # Random starting position in box
            low = -boxSize / 2
            high = boxSize / 2
            startPos = np.random.uniform(low=low, high=high, size=3)

            # Random orientation
            orientation = np.random.uniform(-1, 1, size=3)
            # Uncomment this \/ to set to already aligned perfectly
            #orientation = np.array([1.0,1.0,1.0])
            orientation /= np.linalg.norm(orientation)

            # Generate position of particles along rod
            rodPositions = [startPos + i * rodSpacing * orientation for i in range(rodLength)]
            rodPositions = np.array(rodPositions)
    

            # Check if all positions are inside the box
            if np.all((rodPositions >= low) & (rodPositions <= high)):
                rods.extend(rodPositions)
                break

    return np.array(rods)



# Parameters for initialisation
Lx, Ly, Lz = 12, 12, 12  # Box sizes
numRods = 40  # Number of rods
rodLength = 5  # Particles per rod
rodSpacing = 1.0  # Distance between rod particles
numSolvent = 2500 # Number of solvent particles

# Parameters for simulation
dt = 0.00025  # Time step
dtWarmup = dt / 50
drivingForceMagnitude = 10 # Magnitude of force driving rods forward
warmupLength = 200  # Number of timesteps to tune forces to prevent extreme initial velocities
simLength = 1000  # Number of timesteps for run of simulation
outStep = 50  # Periodicity of output frames
kBond = 750  # Strength of force between particles in rod
kAngle = 500  # Strength to keep particles in rod aligned
sigma = 2.0  # Range over which leonard jones potentials will stretch
kT = 0.01  # Kinetic energy given to wholes system after warmup
gammaSolvent = 2  # Slight resistance added to solvents
gammaRod = 1  # slight resistance added to rod

# Also parameters for simulation, but the strength of leonard jones potentials
ssei = 0.00001  # solvent solvent epsilon initial
rsei = 0.00005  # rod solvent epsilon initial
rrei = 0.0001  # rod rod epsilon initial
ssef = 0.5  # solvent solvent epsilon final
rsef = 0.5  # rod solvent epsilon final
rref = 1  # rod rod epsilon final

params = {"Lx": Lx, "Ly": Ly, "Lz": Lz, "numRods": numRods, "rodLength": rodLength, "rodSpacing": rodSpacing, "numSolvent": numSolvent, 
        "dt": dt, "dtWarmup": dtWarmup, "drivingForceMagnitude": drivingForceMagnitude, "warmupLength": warmupLength, "simLength": simLength, "outStep": outStep, "kBond": kBond, "kAngle": kAngle, "sigma": sigma, "kT": kT, "gammaSolvent": gammaSolvent, "gammaRod": gammaRod,
        "ssei": ssei, "rsei": rsei, "rrei": rrei, "ssef": ssef, "rsef": rsef, "rref": rref}



with open("simulationMetadata.json", "w") as f:
    json.dump(params, f, indent=4)


# Create the initial configuration
box = hoomd.Box(Lx=Lx, Ly=Ly, Lz=Lz)

# Generate particle positions
solventPositions = np.random.uniform(low=-Lx / 2, high=Lx / 2, size=(numSolvent, 3))
rodPositions = GenerateRandomRods(numRods, rodLength, Lx, rodSpacing)

# Create snapshot
snapshot = hoomd.Snapshot()
snapshot.particles.N = numSolvent + numRods * rodLength
snapshot.particles.types = ['solvent', 'rod']
snapshot.particles.position[:] = np.vstack([solventPositions, rodPositions])
snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]


# Bonds
snapshot.bonds.N = numRods * (rodLength - 1)
snapshot.bonds.types = ['rodBond']
bondList = []
for i in range(numRods):
    startIndex = numSolvent + i * rodLength
    for j in range(rodLength - 1):
        bondList.append([startIndex + j, startIndex + j + 1])
snapshot.bonds.group[:] = bondList
snapshot.bonds.typeid[:] = [0] * len(bondList)

# Update particle types (assign all rod particles the same type)
snapshot.particles.typeid[:] = [0] * numSolvent + [1] * (rodLength * numRods)


# Angles
snapshot.angles.N = numRods * (rodLength - 2)
snapshot.angles.types = ['rodAngle']
angleList = []
for i in range(numRods):
    startIndex = numSolvent + i * rodLength
    for j in range(rodLength - 2):
        angleList.append([startIndex + j, startIndex + j + 1, startIndex + j + 2])
snapshot.angles.group[:] = angleList
snapshot.angles.typeid[:] = [0] * len(angleList)

device = hoomd.device.CPU()
sim = hoomd.Simulation(device=device, seed=42)

sim.create_state_from_snapshot(snapshot)

hoomd.write.GSD.write(state=sim.state, filename="rodsInitial.gsd", mode="wb")


print("Initial GSD file 'rodsInitial.gsd' created with metadata file.")
