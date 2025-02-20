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


# Make png of entropy heatmpa in 3d of a single frame
def HeatmapFrame(entropyFlat, xFlat, yFlat, zFlat, vmin, vmax, frameDir, i):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="3d")

    opacity = np.clip(entropyFlat**2, 0.05, 1)

    sc = ax.scatter(xFlat, yFlat, zFlat, c=entropyFlat, cmap="inferno", alpha=opacity, marker="o", s=10, vmin=vmin, vmax=vmax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(f"Frame {i}")

    cbar = plt.colorbar(sc)
    cbar.set_label("Entropy")

    frameFilename = os.path.join(frameDir, f"frame_{i}.png")
    plt.savefig(frameFilename)
    plt.close(fig)

    return(frameFilename)


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

    return timesteps, types, Lx, Ly, Lz, numSolvent, numRods, rodLength, rodSpacing


def Animate(h5Filename, trajGifFilename, hmGifFilename, Lx, Ly, Lz, timesteps, types, numSolvent, numRods, rodLength):
    # Open the HDF5 file and read the positions
    f = h5py.File(h5Filename, "r")
    
    # Create a directory to store the individual frames for the GIF
    trajFrameDir = "trajFrames"
    hmFrameDir = "hmFrames"
    if not os.path.exists(trajFrameDir):
        os.makedirs(trajFrameDir)
    if not os.path.exists(hmFrameDir):
        os.makedirs(hmFrameDir)
        
    # Will store images
    trajImages = []
    heatmapImages = []

    colourmap = plt.cm.bwr
    entropyAllFrames = []

    # Initialise grid for use by entropy meatmap
    gridSize = 2
    gridX, gridY, gridZ = np.meshgrid(
            np.linspace(-Lx/2, Lx/2, gridSize),
            np.linspace(-Ly/2, Ly/2, gridSize),
            np.linspace(-Lz/2, Lz/2, gridSize))
    xFlat, yFlat, zFlat = gridX.ravel(), gridY.ravel(), gridZ.ravel()
    positionsGrid = np.vstack([xFlat, yFlat, zFlat])

    
    for timestep in timesteps:
        # Extract the particle positions for this timestep
        dataset_name = f"step_{timestep}"
        positions = np.array(f[dataset_name][:])         


        s2, allRodPositions = ComputeOrderParameter(positions, numSolvent, numRods, rodLength)
        # Generatre frame for trajectory
        trajFilename = TrajFrame(positions, types, trajFrameDir, timestep, colourmap, Lx, Ly, Lz, s2, allRodPositions)
        # Calculate entropy in 3d grid of the frame
        entropyAllFrames.append(CalculateEntropy(positions, positionsGrid, gridX.shape))
 

        # Read the image and append it to the images list
        trajImages.append(Image.open(trajFilename))
        
    # Find biggest and largest entropies to use as scale for plotting for entropy heatmap gif
    globalVMin = np.min(entropyAllFrames)
    globalVMax = np.max(entropyAllFrames)
        
    # Go through entropies calculated and generate frames
    for i, entropyFlat in enumerate(entropyAllFrames):
            
        hmFilename = HeatmapFrame(entropyFlat, xFlat, yFlat, zFlat, globalVMin, globalVMax, hmFrameDir, i)
        heatmapImages.append(Image.open(hmFilename))

    # Create a GIF from the images
    trajImages[0].save(trajGifFilename, save_all=True, append_images=trajImages[1:], duration=200, loop=0)  # Duration between frames (milliseconds)
    heatmapImages[0].save(hmGifFilename,save_all=True, append_images=heatmapImages[1:], duration=200, loop=0)

    # Clean up by removing the frame images
    for trajFilename in os.listdir(trajFrameDir):
        os.remove(os.path.join(trajFrameDir, trajFilename))
    os.rmdir(trajFrameDir)
    for hmFilename in os.listdir(hmFrameDir):
        os.remove(os.path.join(hmFrameDir, hmFilename))
    os.rmdir(hmFrameDir)

    print(f"GIFs saved as {trajGifFilename} and {hmGifFilename}")




def main(metadataFilename, h5Filename, trajGifFilename, hmGifFilename):

    timesteps, types, Lx, Ly, Lz, numSolvent, numRods, rodLength, rodSpacing = ExtractData(metadataFilename, h5Filename)

    Animate(h5Filename, trajGifFilename, hmGifFilename, Lx, Ly, Lz, timesteps, types, numSolvent, numRods, rodLength)


# Define input filenames
metadataFilename = "simulationMetadata.json"
h5Filename = "positions.h5"  # HDF5 file containing particle positions
trajGifFilename = "trajectory3d.gif"  # Name of the output GIF file
hmGifFilename = "heatmap3d.gif"

main(metadataFilename, h5Filename, trajGifFilename, hmGifFilename)

