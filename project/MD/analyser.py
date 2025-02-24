import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import sys
import json
from scipy.spatial.distance import pdist



def ComputeOrderParameter(positions, numSolvents, numRods, rodLength):
    rodAxes = np.zeros((numRods, 3))
    allRodPositions = np.zeros((numRods, 3, rodLength))

    for rodNum in range(numRods):
        rodStartTag = numSolvents + rodNum * rodLength
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



def Iterate(h5Filename, timesteps, params):
    # Open the HDF5 file and read the positions
    f = h5py.File(h5Filename, "r")
    
    msdValues = np.zeros(len(timesteps))
    S2Values = np.zeros(len(timesteps))

    index = 0

    for timestep in timesteps:
        # Extract the particle positions for this timestep
        dataset_name = f"step_{timestep}"
        positions = np.array(f[dataset_name][:])         

        [numSolvents, numRods, rodLength] = params["numSolvents"], params["numRods"], params["rodLength"]

        S2, P2Values, allRodPositions = ComputeOrderParameter(positions, numSolvents, numRods, rodLength)
        msd = ComputeMSD(allRodPositions)

        msdValues[index], S2Values[index] = msd, S2

        index += 1

    # Correlation tells if aigned rods move differently from unaligned
    # If correlation > 0, more aligned rods move more, and move less if correlation < 0
    correlation = np.corrcoef(msdValues, S2Values)[0,1]

    return (msdValues, S2Values, correlation)



def Plot(timesteps, msdValues, S2Values, ID):

    fig, ax1 = plt.subplots()

    ax1.plot(timesteps, msdValues, color="blue", alpha=0.6, label="Mean Squared Distance")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Squared Displacement", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(timesteps, S2Values, color="red", alpha=0.6, label="Order Parameter")
    ax2.set_ylabel("Order Parameter", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("MSD and Order Parameter over Time")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    
    plt.show()
    plt.savefig(f"MSDvsS2{ID}.png")



def ViewLog(logFilename, ID):

    hdf5File = h5py.File(name=logFilename, mode="r")

    timestep = hdf5File["hoomd-data/Simulation/timestep"][:]
    potential_energy = hdf5File["hoomd-data/md/compute/ThermodynamicQuantities/potential_energy"][:]

    print("Available quantities: {'kinetic_temperature': 'scalar', 'pressure': 'scalar', 'pressure_tensor': 'sequence', 'kinetic_energy': 'scalar', 'translational_kinetic_energy': 'scalar', 'rotational_kinetic_energy': 'scalar', 'potential_energy': 'scalar', 'degrees_of_freedom': 'scalar', 'translational_degrees_of_freedom': 'scalar', 'rotational_degrees_of_freedom': 'scalar', 'num_particles': 'scalar', 'volume': 'scalar'}")

    plt.plot(timestep, potential_energy)
    plt.xlabel("timestep")
    plt.ylabel("potential energy")
    plt.savefig(f"outLog{ID}.png")

    print(f"Saved figure to outLog{ID}.py")



def main(metadataFilename, h5Filename, logFilename, ID):

    timesteps, params = ExtractData(metadataFilename, h5Filename)

    msdValues, S2Values, correlation = Iterate(h5Filename, timesteps, params)

    print(f"Correlation: {correlation}")

    Plot(timesteps, msdValues, S2Values, ID)

    ViewLog(logFilename, ID)


inputs = sys.argv
ID = ""  # Default to blank

if (len(inputs) > 1):
    inp = inputs[1]  # Assume ID given by first argument, ignore rest
    if ("=" in inp):
        param, value = inp.split("=")[0], inp.split("=")[1]
        if (param == "ID"):
            ID = value


# Define input filenames
metadataFilename = f"simulationMetadata{ID}.json"
h5Filename = f"positions{ID}.h5"  # HDF5 file containing particle positions
logFilename = f"log{ID}.h5"

main(metadataFilename, h5Filename, logFilename, ID)

