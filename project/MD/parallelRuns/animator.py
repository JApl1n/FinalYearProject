import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import sys
import json
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist



def ComputeMSD(allRodPositions):

    rodParticles = allRodPositions.transpose(0,2,1).reshape(-1,3)

    distances = pdist(rodParticles, metric="euclidean")
    msd = np.mean(distances**2)

    return msd


# Extend positions by mirroring across periodic boundaries.
# This means probability density calculated accounts for periodic boundary conditions
def ApplyPBC(positions, boxSize):
    [Lx, Ly, Lz] = boxSize
    mirroredPositions = [positions]
    shifts = [-1, 0, 1]  # Shift in each dimension

    for dx in shifts:
        for dy in shifts:
            for dz in shifts:
                if dx==dy==dz==0:
                    continue
                shiftVector = np.array([dx*Lx, dy*Ly, dz*Lz])
                mirroredPositions.append(positions + shiftVector)

    return np.vstack(mirroredPositions)



# Calculate entropy of a single frame. This is related to Shannon entropy, measuring uncertainty or disorder
# in a probability distribution. We use a discrete approximation of the differential entropy.
def CalculateEntropy(allRodPositions, positionsGrid, gridShape, boxSize):
    positions = allRodPositions.transpose(0,2,1).reshape(-1,3)

    extendedPositions = ApplyPBC(positions, boxSize)

    kde = gaussian_kde(extendedPositions.T, bw_method = "scott")
    
    density = kde(positionsGrid).reshape(gridShape)

    entropy = -density * np.log(density + 1e-10)
    entropy = entropy / np.max(entropy)

    return(entropy.ravel())


# Make png of entropy heatmpa in 3d of a single frame
def HeatmapFrame(entropyFlat, xFlat, yFlat, zFlat, vmin, vmax, frameDir, i):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="3d")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    opacity = np.clip((entropyFlat-(entropyFlat.min()*0.7)), 0.01, 0.75)
    
    colourmap = plt.cm.inferno
    colours = colourmap(norm(entropyFlat))
    colours[:, 3] = opacity

    sc = ax.scatter(xFlat, yFlat, zFlat, c=colours,  marker="o", s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(f"Frame {i}")

    sm = cm.ScalarMappable(norm=norm, cmap=colourmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Relative Shannon Entropy")

    frameFilename = os.path.join(frameDir, f"frame_{i}.png")
    plt.savefig(frameFilename)
    plt.close(fig)

    return(frameFilename)


# Make a trajectory png for a single frame in 3d
def TrajFrame(positions, types, frameDir, timestep, colourmap, params, s2, msd, allRodPositions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    Lx, Ly, Lz = params["Lx"], params["Ly"], params["Lz"]

    # Extract x, y, z positions
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    normalisedTypes = types / np.max(types)  # Normalize tags to fit in the colormap range
    
    # Assign colors based on the normalized tags
    colours = colourmap(normalisedTypes)
    colours[:,3] = (normalisedTypes*0.67)+0.33  # Set alpha values
    
    sizes = (normalisedTypes*20)+10

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
                    linewidth=2, color="black", alpha=0.5)

        #ax.plot(rodPositions[0], rodPositions[1], rodPositions[2], linewidth=2, color="black")

    # Set plot limits
    ax.set_xlim([-Lx/2, Lx/2])
    ax.set_ylim([-Ly/2, Ly/2])
    ax.set_zlim([-Lz/2, Lz/2])

    # Labels for the axes
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")

    

    ax.set_title(f"Order Parameter: {round(s2, 4)}, MSD: {round(msd, 2)}", size=10)

    # Save the current plot as an image
    frameFilename = os.path.join(frameDir, f"frame_{timestep}.png")
    plt.savefig(frameFilename)
    plt.close(fig)

    return(frameFilename)



def ComputeOrderParameter(positions, params):
    numSolvents, numRods, rodLength = params["numSolvents"], params["numRods"], params["rodLength"]

    rodAxes = np.zeros((numRods, 3))
    allRodPositions = np.zeros((numRods, 3, rodLength))

    for rodNum in range(numRods):
        rodStartTag = numSolvents + rodNum * rodLength
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
        #print("Available keys in HDF5 file:", f.keys())
        # Get all available time steps (dataset names)
        timesteps = sorted([int(step.split('_')[-1]) for step in f.keys() if metaName not in step])

        types = f[metaName][:]
    f.close()

    with open(metadataFilename, "r") as f:
        params = json.load(f)
    f.close()

    return timesteps, types, params



def Animate(h5Filename, trajGifFilename, hmGifFilename, timesteps, types, params, ID):
    # Open the HDF5 file and read the positions
    f = h5py.File(h5Filename, "r")
    
    # Create a directory to store the individual frames for the GIF
    trajFrameDir = f"trajFrames{ID}"
    hmFrameDir = f"hmFrames{ID}"
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
    gridSize = 5
    Lx, Ly, Lz = params["Lx"], params["Ly"], params["Lz"]
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


        s2, allRodPositions = ComputeOrderParameter(positions, params)
        # Compute mean squared distance
        msd =  ComputeMSD(allRodPositions)
        # Generatre frame for trajectory
        trajFilename = TrajFrame(positions, types, trajFrameDir, timestep, colourmap, params, s2, msd, allRodPositions)
        # Calculate entropy in 3d grid of the frame
        entropyAllFrames.append(CalculateEntropy(allRodPositions, positionsGrid, gridX.shape, np.array([Lx, Ly, Lz])))
 

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




def main(metadataFilename, h5Filename, trajGifFilename, hmGifFilename, ID):

    timesteps, types, params = ExtractData(metadataFilename, h5Filename)

    Animate(h5Filename, trajGifFilename, hmGifFilename, timesteps, types, params, ID)


inputs = sys.argv
ID = ""  # Default to blank

if (len(inputs) > 1):
    inp = inputs[1]  # Assume ID given by first argument, ignore rest
    if ("=" in inp):
        param, value = inp.split("=")[0], inp.split("=")[1]
        if (param == "ID"):
            ID = value


# Define input filenames
metadataFilename = f"metadata/simulationMetadata{ID}.json"
h5Filename = f"positions/positions{ID}.h5"  # HDF5 file containing particle positions
trajGifFilename = f"outGifs/trajectory3d{ID}.gif"  # Name of the output GIF file
hmGifFilename = f"outGifs/heatmap3d{ID}.gif"

main(metadataFilename, h5Filename, trajGifFilename, hmGifFilename, ID)

