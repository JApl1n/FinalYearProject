import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

import json
from scipy.spatial.distance import pdist



def ComputeOrderParameter(positions, numSolvent, numRods, rodLength):
    rodAxes = np.zeros((numRods, 3))
    allRodPositions = np.zeros((numRods, 3, rodLength))

    for rodNum in range(numRods):
        rodStartTag = numSolvent + rodNum * rodLength
        rodEndTag = rodStartTag + rodLength

        rodPositions = positions[rodStartTag:rodEndTag]
        allRodPositions[rodNum] = rodPositions.T
        rodAxis = rodPositions[-1] - rodPositions[0]

        rodAxis /= np.linalg.norm(rodAxis)  # Normalise rod axis
        rodAxes[rodNum] = rodAxis

    # Compute director, mean rod direction
    director = np.mean(rodAxes, axis=0)
    director /= np.linalg.norm(director)  
    
    # Compute cos^2(theta) for each rod
    cosThetaSquared = (rodAxes @ director) ** 2

    # Compute second order polynomial for each rod
    P2Values = (3/2) * cosThetaSquared - 0.5

    # Compute overall order parameter
    S2 = np.round(np.mean(P2Values), 6)

    return(S2, P2Values, allRodPositions)



def ComputeMSD(allRodPositions):

    rodParticles = allRodPositions.transpose(0,2,1).reshape(-1,3)
    
    distances = pdist(rodParticles, metric="euclidean")
    msd = np.mean(distances**2)

    return msd



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

    return timesteps, params



def Iterate(h5Filename, trajGifFilename, hmGifFilename, timesteps, params):
    # Open the HDF5 file and read the positions
    f = h5py.File(h5Filename, "r")
    
    msdValues = np.zeros(len(timesteps))
    S2Values = np.zeros(len(timesteps))

    index = 0

    for timestep in timesteps:
        # Extract the particle positions for this timestep
        dataset_name = f"step_{timestep}"
        positions = np.array(f[dataset_name][:])         

        [numSolvent, numRods, rodLength] = params["numSolvent"], params["numRods"], params["rodLength"]

        S2, P2Values = allRodPositions = ComputeOrderParameter(positions, numSolvent, numRods, rodLength)
        msd = ComputeMSD(allRodPositions)

        msdValues[index], S2Values[index] = msd, S2

        index += 1

    # Correlation tells if aigned rods move differently from unaligned
    # If correlation > 0, more aligned rods move more, and move less if correlation < 0
    correlation = np.corrcoef(msdValues, S2Values)[0,1]

    return (msdValues, S2Values, correlation)



def Plot(timesteps, msdValues, S2Values):

    plt.plot(timesteps, msdValues, color="blue", alpha=0.6)
    plt.xlabel("Timestep")
    plt.ylabel("Mean Squared Displacement")
    plt.title("MSD over time")
    plt.show()
    plt.savefig("MSD.png")
    plt.close()

    plt.plot(timesteps, S2Values, color="blue", alpha=0.6)
    plt.xlabel("Timestep")
    plt.ylabel("Order Parameter")
    plt.title("Order Parameter over time")
    plt.show()
    plt.savefig("S2.png")



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

    msdValues, S2Values, correlation= Iterate(h5Filename, trajGifFilename, hmGifFilename, timesteps, params)

    print(f"Correlation: {correlation}")

    Plot(timesteps, msdValues, S2Values)

    ViewLog(logFilename)


# Define input filenames
metadataFilename = "simulationMetadata.json"
h5Filename = "positions.h5"  # HDF5 file containing particle positions
trajGifFilename = "trajectory3d.gif"  # Name of the output GIF file
hmGifFilename = "heatmap3d.gif"
logFilename = "log.h5"

main(metadataFilename, h5Filename, trajGifFilename, hmGifFilename, logFilename)

