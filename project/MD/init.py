import hoomd
import numpy as np
import json


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


# Parameters
Lx, Ly, Lz = 20, 20, 20  # Box dimensions
numRods = 50  # Number of rods
rodLength = 5  # Particles per rod
rodSpacing = 1.0  # Distance between rod particles
numSolvent = 500 # Number of solvent particles

params = {"Lx": Lx, "Ly": Ly, "Lz": Lz, "numRods": numRods, "rodLength": rodLength, "rodSpacing": rodSpacing, "numSolvent": numSolvent}

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
