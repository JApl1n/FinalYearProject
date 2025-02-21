import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

import json
from scipy.stats import gaussian_kde

# orientational order parameter-  use it
# second order legendre polynomial, find preferred direction 
# Calculate entropy of a single frame
def CalculateEntropy(positions, positionsGrid, shape):
    kde = gaussian_kde(positions.T, bw_method = "scott")
    
    density = kde(positionsGrid).reshape(shape)

    entropy = -density * np.log(density + 1e-10)
    entropy = entropy / np.max(entropy)

    return(entropy.ravel())



# Make a trajectory png for a single frame in 3d
def TrajFrame(positions, types, frameDir, timestep, colourmap, Lx, Ly, Lz, s2, allRodPositions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z positions
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    normalisedTypes = types / np.max(types)  # Normalize tags to fit in the colormap range
    
    # Assign colors based on the normalized tags
    colours = colourmap(normalisedTypes)
    sizes = normalisedTypes*12 + 4


    # Scatter plot for positions
    ax.scatter(x, y, z, s=sizes, c=colours, marker="o")
    for rodPositions in allRodPositions:
        xRod, yRod, zRod = rodPositions

        dx = np.abs(np.diff(xRod))
        dy = np.abs(np.diff(yRod))
        dz = np.abs(np.diff(zRod))

        mask = (dx < Lx/2) & (dy < Ly/2) & (dz < Lz/2)
        validIndices = np.where(mask)[0]

        for idx in validIndices:
            ax.plot(
                    [xRod[idx], xRod[idx+1]],
                    [yRod[idx], yRod[idx+1]],
                    [zRod[idx], zRod[idx+1]],
                    linewidth=2, color="black")

        #ax.plot(rodPositions[0], rodPositions[1], rodPositions[2], linewidth=2, color="black")

    # Set plot limits
    ax.set_xlim([-Lx/2, Lx/2])
    ax.set_ylim([-Ly/2, Ly/2])
    ax.set_zlim([-Lz/2, Lz/2])

    # Labels for the axes
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")

    ax.set_title(f"Order Parameter: {s2}", size=10)

    # Save the current plot as an image
    frameFilename = os.path.join(frameDir, f"frame_{timestep}.png")
    plt.savefig(frameFilename)
    plt.close(fig)

    return(frameFilename)



def ComputeOrderParameter(positions, numSolvent, numRods, rodLength):
    rodAxes = np.zeros((numRods, 3))
    allRodPositions = np.zeros((numRods, 3, rodLength))
    for rodNum in range(numRods):
        rodStartTag = numSolvent + rodNum * rodLength
        rodEndTag = rodStartTag + rodLength

        rodPositions = positions[rodStartTag:rodEndTag]
        allRodPositions[rodNum] = rodPositions.T
        rodAxis = rodPositions[-1] - rodPositions[0]

        rodAxis /= np.linalg.norm(rodAxis)
        rodAxes[rodNum] = rodAxis

    director = np.mean(rodAxes, axis=0)
    director /= np.linalg.norm(director)
    
    cosThetaSquared = (rodAxes @ director) **2
    s2 = np.round((3/2) * np.mean(cosThetaSquared) - 0.5, 6)

    return(s2, allRodPositions)


def ExtractData(metadataFilename, h5Filename):
    metaName = "metadata"
    with h5py.File(h5Filename, "r") as f:
        print("Available keys in HDF5 file:", f.keys())
        # Get all available time steps (dataset names)
        timesteps = sorted([int(step.split('_')[-1]) for step in f.keys() if metaName not in step])

        types = f[metaName][:]
    f.close()

    with open(metadataFilename, "r") as f:
        params = json.load(f)
    f.close()

    Lx, Ly, Lz = params["Lx"], params["Ly"], params["Lz"]
    numSolvent = params["numSolvent"]
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

    return timesteps, params


def Animate(h5Filename, trajGifFilename, hmGifFilename, timesteps, params):
    # Open the HDF5 file and read the positions
    f = h5py.File(h5Filename, "r")
    
    for timestep in timesteps:
        # Extract the particle positions for this timestep
        dataset_name = f"step_{timestep}"
        positions = np.array(f[dataset_name][:])         

        [numSolvent, numRods, rodLength] = params["numSolvent"], params["numRods"], params["rodLength"]

        s2, allRodPositions = ComputeOrderParameter(positions, numSolvent, numRods, rodLength)
        # Generatre frame for trajectory
        #trajFilename = TrajFrame(positions, types, trajFrameDir, timestep, colourmap, Lx, Ly, Lz, s2, allRodPositions)
        # Calculate entropy in 3d grid of the frame
        #entropyAllFrames.append(CalculateEntropy(positions, positionsGrid, gridX.shape))
 
                


def ViewLog(logFilename):

    hdf5File = h5py.File(name=logFilename, mode="r")

    timestep = hdf5File["hoomd-data/Simulation/timestep"][:]
    potential_energy = hdf5File["hoomd-data/md/compute/ThermodynamicQuantities/potential_energy"][:]

    print("Available quantities: {'kinetic_temperature': 'scalar', 'pressure': 'scalar', 'pressure_tensor': 'sequence', 'kinetic_energy': 'scalar', 'translational_kinetic_energy': 'scalar', 'rotational_kinetic_energy': 'scalar', 'potential_energy': 'scalar', 'degrees_of_freedom': 'scalar', 'translational_degrees_of_freedom': 'scalar', 'rotational_degrees_of_freedom': 'scalar', 'num_particles': 'scalar', 'volume': 'scalar'}")

    plt.plot(timestep, potential_energy)
    plt.xlabel("timestep")
    plt.ylabel("potential energy")
    plt.savefig("outLog.png")

    print("Saved figure to outLog.py")



def main(metadataFilename, h5Filename, trajGifFilename, hmGifFilename, logFilename):

    timesteps, params = ExtractData(metadataFilename, h5Filename)

    Animate(h5Filename, trajGifFilename, hmGifFilename, timesteps, params)

    ViewLog(logFilename)

# Define input filenames
metadataFilename = "simulationMetadata.json"
h5Filename = "positions.h5"  # HDF5 file containing particle positions
trajGifFilename = "trajectory3d.gif"  # Name of the output GIF file
hmGifFilename = "heatmap3d.gif"
logFilename = "log.h5"

main(metadataFilename, h5Filename, trajGifFilename, hmGifFilename, logFilename)

