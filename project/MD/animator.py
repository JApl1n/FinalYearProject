import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

from scipy.stats import gaussian_kde

def HeatmapFrame(positions, frameDir, timestep, Lx, Ly, Lz):

    gridSize = 30
    kde = gaussian_kde(positions.T, bw_method = "scott")
    gridX, gridY, gridZ = np.meshgrid(
        np.linspace(-Lx/2, Lx/2, gridSize),
        np.linspace(-Ly/2, Ly/2, gridSize),
        np.linspace(-Lz/2, Lz/2, gridSize))

    xFlat, yFlat, zFlat = gridX.ravel(), gridY.ravel(), gridZ.ravel()
    positionsGrid = np.vstack([xFlat, yFlat, zFlat])
    density = kde(positionsGrid).reshape(gridX.shape)

    # Compute entropy in 3D
    entropy = -density * np.log(density + 1e-10)
    entropy = entropy / np.max(entropy)

    entropyFlat = entropy.ravel()

    opacity = np.clip(entropyFlat ** 2, 0.05, 1)

    # Visualisation using
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(xFlat, yFlat, zFlat, c=entropyFlat, cmap="inferno", alpha=opacity, marker="o", s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Entropy Contour")

    plt.colorbar(sc, label="entropy")
    frameFilename = os.path.join(frameDir, f"frame_{timestep}.png")
    plt.savefig(frameFilename)
    plt.close

    return(frameFilename)


def TrajFrame(positions, types, frameDir, timestep, colourmap, Lx, Ly, Lz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z positions
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    normalisedTypes = types / np.max(types)  # Normalize tags to fit in the colormap range
    # Assign colors based on the normalized tags
    colours = colourmap(normalisedTypes)

    # Scatter plot for positions
    ax.scatter(x, y, z, s=10, c=colours, marker="o")
    # Set plot limits
    ax.set_xlim([-Lx/2, Lx/2])
    ax.set_ylim([-Ly/2, Ly/2])
    ax.set_zlim([-Lz/2, Lz/2])

    # Labels for the axes
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")

    # Save the current plot as an image
    frameFilename = os.path.join(frameDir, f"frame_{timestep}.png")
    plt.savefig(frameFilename)
    plt.close(fig)

    return(frameFilename)


def Animate(h5Filename, trajGifFilename, hmGifFilename, Lx, Ly, Lz):
    # Open the HDF5 file and read the positions
    with h5py.File(h5Filename, "r") as f:
        print("Available keys in HDF5 file:", f.keys())
        # Get all available time steps (dataset names)
        timesteps = sorted([int(step.split('_')[-1]) for step in f.keys() if "types" not in step])

        types = f["types"][:]
        
        # Create a directory to store the individual frames for the GIF
        trajFrameDir = "trajFrames"
        hmFrameDir = "hmFrames"
        if not os.path.exists(trajFrameDir):
            os.makedirs(trajFrameDir)
        if not os.path.exists(hmFrameDir):
            os.makedirs(hmFrameDir)
        
        # Collect the images for the GIF
        trajImages = []
        heatmapImages = []

        colourmap = plt.cm.bwr
        colourbar = False

        for timestep in timesteps:
            # Extract the particle positions for this timestep
            dataset_name = f"step_{timestep}"
            positions = f[dataset_name][:]          
            
            trajFilename = TrajFrame(positions, types, trajFrameDir, timestep, colourmap, Lx, Ly, Lz)
            hmFilename = HeatmapFrame(positions, hmFrameDir, timestep, Lx, Ly, Lz)
 

            # Read the image and append it to the images list
            trajImages.append(Image.open(trajFilename))
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



        print(f"GIF saved as {trajGifFilename}")

# Example usage:
h5Filename = "positions.h5"  # Your HDF5 file containing particle positions
trajGifFilename = "trajectory3d.gif"  # Name of the output GIF file
hmGifFilename = "heatmap3d.gif"

Lx, Ly, Lz = 20, 20, 20

Animate(h5Filename, trajGifFilename, hmGifFilename, Lx, Ly, Lz)
