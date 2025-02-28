import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



def LoadData(inputFolder):
    allIDs = []
    allParams = []
    allMSDVals = []
    allS2Vals = []
    allCorrelations = []

    # Load data also used by using every file in folder
    for filename in os.listdir(inputFolder):
        with open(os.path.join(inputFolder, filename), "r") as f:
            data = json.load(f)
   
        allIDs.append(data["ID"])
        allParams.append(data["params"])
        allMSDVals.append(data["msdValues"])
        allS2Vals.append(data["S2Values"])
        allCorrelations.append(data["correlation"])

    allIDs = np.array(allIDs)
    allMSDVals = np.array(allMSDVals)
    allS2Vals = np.array(allS2Vals)
    allCorrelations = np.array(allCorrelations)

    return(allIDs, allParams, allMSDVals, allS2Vals, allCorrelations)
    


def LoadLogs(inputFolder, quantity):
    timesteps = []
    quantities = []

    for filename in os.listdir(inputFolder):
        hdf5File = h5py.File(name=f"{inputFolder}/{filename}", mode="r")
            
        timesteps.append(hdf5File["hoomd-data/Simulation/timestep"][:])
        quantities.append(hdf5File[f"hoomd-data/md/compute/ThermodynamicQuantities/{quantity}"][:])

    return(timesteps, quantities)
        


# Find differing parameters across simulations
def Diff(allParams):
    paramVals = defaultdict(set)
 
    paramLabels = allParams[0].keys()
    for sim in allParams:
        for key, value in sim.items():
            paramVals[key].add(value)

    varyingParams = {key: values for key, values in paramVals.items() if len(values) > 1}

    diffKeys = []
    diffVals = []

    if varyingParams:
        print("The following parameters vary between simulations:")
        for key, values in varyingParams.items():
            print(f"{key}: {values}")
            diffKeys.append(key)
            diffVals.append(value)
    else:
        print("All parameters are the same across simulations.")

    return(diffKeys, diffVals)



def Plot(allIDs, allParams, allMSDVals, allS2Vals, allCorrelations, folderName):
    order = np.argsort(allIDs)

    plt.violinplot(allS2Vals[order].T)
    plt.xticks(np.arange(1, len(allIDs)+1), labels=np.arange(3,9))
    plt.xlabel("Rod Length")
    plt.ylabel("Order Parameter")
    plt.title("How does the Order Parameter vary with changing Rod Length")
    plt.show()
    plt.savefig(f"{folderName}/simComparison.png")
    plt.close()



def MSDvsS2(timesteps, msdValues, S2Values, correlation, ID, folderName):
    fig, ax1 = plt.subplots()

    fig.set_figheight(8)
    fig.set_figwidth(15)

    ax1.plot(timesteps, msdValues, color="blue", alpha=0.6, label="Mean Squared Distance")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Squared Displacement", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(timesteps, S2Values, color="red", alpha=0.6, label="Order Parameter")
    ax2.set_ylabel("Order Parameter", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(f"MSD and Order Parameter over Time - Correlation: {round(correlation,4)}")
    fig.legend(loc="upper left", bbox_to_anchor=(0.125, 0.88))

    plt.grid()
    plt.show()
    plt.savefig(f"{folderName}/MSDvsS2{ID}.png")
    plt.close()



def ViewLog(folderName, timestep, quantityValues, quantity, ID):
    
    plt.plot(timestep, quantityValues)
    plt.xlabel("Timestep")
    plt.ylabel("Kinetic Energy")
    plt.title(f"How does kinetic energy evolve for the system over time?")
    plt.grid()
    plt.show()
    plt.savefig(f"outPngs/outLog{ID}.png")
    plt.close()

    print(f"Saved figure to outLog{ID}.py")



def main(inputFolder, logFolder, outputFolder):

    allIDs, allParams, allMSDVals, allS2Vals, allCorrelations = LoadData(inputFolder)
    diffKeys, diffVals = Diff(allParams)

    #print("Available log quantities: {'kinetic_temperature': 'scalar', 'pressure': 'scalar', 'pressure_tensor': 'sequence', 'kinetic_energy': 'scalar', 'translational_kinetic_energy': 'scalar', 'rotational_kinetic_energy': 'scalar', 'potential_energy': 'scalar', 'degrees_of_freedom': 'scalar', 'translational_degrees_of_freedom': 'scalar', 'rotational_degrees_of_freedom': 'scalar', 'num_particles': 'scalar', 'volume': 'scalar'}")
    quantity = "kinetic_energy"
    timesteps, logQuantities = LoadLogs(logFolder, quantity)
    
    Plot(allIDs, allParams, allMSDVals, allS2Vals, allCorrelations, outputFolder)
    
    index = 0
    for ID in allIDs:
        MSDvsS2(timesteps[index], allMSDVals[index], allS2Vals[index], allCorrelations[index], ID, outputFolder)
        ViewLog(outputFolder, timesteps[index], logQuantities[index], quantity, ID)

        index += 1



inputFolder = "multiSimData"
logFolder = "logs" 
outputFolder = "outPngs"

main(inputFolder, logFolder, outputFolder)



