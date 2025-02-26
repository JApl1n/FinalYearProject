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
Lx, Ly, Lz = 16, 16, 16  # Box sizes
numRods = 100  # Number of rods
rodLength = 5  # Particles per rod
rodSpacing = 0.75  # Distance between rod particles
numSolvents = 4000 # Number of solvent particles

# Parameters for simulation
dt = 0.000025  # Time step
dtWarmup = dt / 50
drivingForceMagnitude = 10 # Magnitude of force driving rods forward
warmupLength = 1000  # Number of timesteps to tune forces to prevent extreme initial velocities
simLength = 4000  # Number of timesteps for run of simulation
outStep = 50  # Periodicity of output frames
kBond = 1250  # Strength of force between particles in rod
kAngle = 750  # Strength to keep particles in rod aligned
sigma = 2.0  # Range over which leonard jones potentials will stretch
kT = 0.01  # Kinetic energy given to whole system after warmup
gammaSolvent = 2.0  # Slight resistance added to solvents
gammaRod = 1.5  # slight resistance added to rod

# Also parameters for simulation, but the strength of leonard jones potentials
ssei = 0.00001  # solvent solvent epsilon initial
rsei = 0.00005  # rod solvent epsilon initial
rrei = 0.0001  # rod rod epsilon initial
ssef = 0.05  # solvent solvent epsilon final
rsef = 0.1  # rod solvent epsilon final
rref = 1.0  # rod rod epsilon final

ID = ""  # An identifier able to differentiate between nodes when multiple simulations ran in parallel

params = {"Lx": Lx, "Ly": Ly, "Lz": Lz, "numRods": numRods, "rodLength": rodLength, "rodSpacing": rodSpacing, "numSolvents": numSolvents, 
        "dt": dt, "dtWarmup": dtWarmup, "drivingForceMagnitude": drivingForceMagnitude, "warmupLength": warmupLength, "simLength": simLength, "outStep": outStep, "kBond": kBond, "kAngle": kAngle, "sigma": sigma, "kT": kT, "gammaSolvent": gammaSolvent, "gammaRod": gammaRod,
        "ssei": ssei, "rsei": rsei, "rrei": rrei, "ssef": ssef, "rsef": rsef, "rref": rref,
        "ID": ID} 



inputs = sys.argv

if (len(inputs) > 1):
    for inp in inputs[1:]:
        if ("=" in inp):
            param, value = inp.split("=")[0], inp.split("=")[1]
            if param in params:
                paramType = type(params[param])
                
                try:
                    old = params[param]

                    value = paramType(value)
                    params[param] = value
                    exec(f"{param} = {value}")
                    print(f"Changing default value of {param} from '{old}' to '{value}'.")
                except ValueError:
                    print(f"There is a datatype mismatch for {param}. Please enter a value of type {paramType}. Using default vlaue instead.")

            else:
                print(f"{param} not in available parameters to change, keeping defaults")
        else:
            print("Invalid entry: use python init.py {parameter name}={new value} {parameter name}={new value}")





with open(f"metadata/simulationMetadata{ID}.json", "w") as f:
    json.dump(params, f, indent=4)


# Create the initial configuration
box = hoomd.Box(Lx=Lx, Ly=Ly, Lz=Lz)

# Generate particle positions
solventPositions = np.random.uniform(low=-Lx / 2, high=Lx / 2, size=(numSolvents, 3))
rodPositions = GenerateRandomRods(numRods, rodLength, Lx, rodSpacing)

# Create snapshot
snapshot = hoomd.Snapshot()
snapshot.particles.N = numSolvents + numRods * rodLength
snapshot.particles.types = ['solvent', 'rod']
snapshot.particles.position[:] = np.vstack([solventPositions, rodPositions])
snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]


# Bonds
snapshot.bonds.N = numRods * (rodLength - 1)
snapshot.bonds.types = ['rodBond']
bondList = []
for i in range(numRods):
    startIndex = numSolvents + i * rodLength
    for j in range(rodLength - 1):
        bondList.append([startIndex + j, startIndex + j + 1])
snapshot.bonds.group[:] = bondList
snapshot.bonds.typeid[:] = [0] * len(bondList)

# Update particle types (assign all rod particles the same type)
snapshot.particles.typeid[:] = [0] * numSolvents + [1] * (rodLength * numRods)


# Angles
snapshot.angles.N = numRods * (rodLength - 2)
snapshot.angles.types = ['rodAngle']
angleList = []
for i in range(numRods):
    startIndex = numSolvents + i * rodLength
    for j in range(rodLength - 2):
        angleList.append([startIndex + j, startIndex + j + 1, startIndex + j + 2])
snapshot.angles.group[:] = angleList
snapshot.angles.typeid[:] = [0] * len(angleList)

device = hoomd.device.CPU()
sim = hoomd.Simulation(device=device, seed=42)

sim.create_state_from_snapshot(snapshot)

hoomd.write.GSD.write(state=sim.state, filename=f"rodsInitial/rodsInitial{ID}.gsd", mode="wb")


print(f"Initial GSD file 'rodsInitial{ID}.gsd' created with metadata file.")
