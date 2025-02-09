import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

def plot_positions_3d(h5_filename, gif_filename):
    # Open the HDF5 file and read the positions
    with h5py.File(h5_filename, "r") as f:
        print("Available keys in HDF5 file:", f.keys())
        # Get all available time steps (dataset names)
        timesteps = sorted([int(step.split('_')[-1]) for step in f.keys() if 'types' not in step])

        types = f["types"][:]
    
        
        # Create a directory to store the individual frames for the GIF
        frame_dir = "frames"
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        
        # Collect the images for the GIF
        images = []

        colourmap = plt.cm.bwr

        for timestep in timesteps:
            # Extract the particle positions for this timestep
            dataset_name = f"step_{timestep}"
            positions = f[dataset_name][:]
            
            # Create the 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Extract x, y, z positions
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]


            normalisedTypes = types / np.max(types)  # Normalize tags to fit in the colormap range
# Assign colors based on the normalized tags
            colours = colourmap(normalisedTypes)

            # Scatter plot for positions
            ax.scatter(x, y, z, s=10, c=colours, marker='o')

            # Set plot limits
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])

            # Labels for the axes
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')

            # Save the current plot as an image
            frame_filename = os.path.join(frame_dir, f"frame_{timestep}.png")
            plt.savefig(frame_filename)
            plt.close(fig)

            # Read the image and append it to the images list
            images.append(imageio.imread(frame_filename))

        # Create a GIF from the images
        imageio.mimsave(gif_filename, images, duration=0.1)  # Duration between frames (seconds)

        # Clean up by removing the frame images
        for frame_filename in os.listdir(frame_dir):
            os.remove(os.path.join(frame_dir, frame_filename))
        os.rmdir(frame_dir)

        print(f"GIF saved as {gif_filename}")

# Example usage:
h5_filename = "positions.h5"  # Your HDF5 file containing particle positions
gif_filename = "particles_3d.gif"  # Name of the output GIF file

plot_positions_3d(h5_filename, gif_filename)
