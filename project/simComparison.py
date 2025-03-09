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
    


def LoadLogs(inputFolder, quantity, IDs):
    foundIDs = []
    timesteps = []
    quantities = []
    

    for filename in os.listdir(inputFolder):
        foundIDs.append(filename.split(".")[0][3:])
        hdf5File = h5py.File(name=f"{inputFolder}/{filename}", mode="r")
            
        currentTimesteps = hdf5File["hoomd-data/Simulation/timestep"][:]
        timesteps.append(currentTimesteps-currentTimesteps.min())
        quantities.append(hdf5File[f"hoomd-data/md/compute/ThermodynamicQuantities/{quantity}"][:])

    # Fixed bug here where the IDs read by reading the logs isnt the same order as everywhere else, so changed them to match here:

    mapper = np.array(IDs, dtype=int)
    timesteps = np.array(timesteps)
    quantities = np.array(quantities)
    timesteps = timesteps[np.argsort(foundIDs)]
    quantities = quantities[np.argsort(foundIDs)]
    timesteps = timesteps[mapper]
    quantities = quantities[mapper]

    
    
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


# Compare MSD or order parameter for all simulations with varying parameter
def ComparePlot(allIDs, allParams, allMSDVals, allS2Vals, allCorrelations, folderName):
    order = np.argsort(allIDs)

    plt.violinplot(allS2Vals[order].T)
    # For now im manually editing the labels instead of using the parameter that is being changed and its value
    plt.xticks(np.arange(1, len(allIDs)+1), labels=np.arange(16,21,1))
    plt.xlabel("numRods")
    plt.ylabel("Order Parameter")
    plt.title("How does the Order Parameter vary with changing numRods")
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


# View on ethermodynamic property over time
def ViewLog(folderName, timestep, quantityValues, quantity, ID):
    plt.plot(timestep, quantityValues)
    plt.xlabel("Timestep")
    plt.ylabel(quantity)
    plt.title(f"How does {quantity} evolve for the system over time?")
    plt.grid()
    plt.show()
    plt.savefig(f"outPngs/outLog{ID}.png")
    plt.close()

    print(f"Saved figure to outLog{ID}.py")


# Compare how two thermodynamic properties evolve over time
def ViewLogs(folderName, timesteps, quantityValues1, quantityValues2, quantity1, quantity2, ID):

    fig, ax1 = plt.subplots()

    fig.set_figheight(7)
    fig.set_figwidth(10)

    ax1.plot(timesteps, quantityValues1, color="blue", alpha=0.6, label=quantity1)
    plt.xlabel("Timestep")
    plt.ylabel(quantity1, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(timesteps, quantityValues2, color="red", alpha=0.6, label=quantity2)
    ax2.set_ylabel(quantity2, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(f"Evolution of {quantity1} and {quantity2} over time ")
    fig.legend(loc="upper left", bbox_to_anchor=(0.125, 0.88))

    plt.grid()
    plt.show()
    plt.savefig(f"outPngs/outLog{ID}.png")
    plt.close()



def main(inputFolder, logFolder, outputFolder):

    allIDs, allParams, allMSDVals, allS2Vals, allCorrelations = LoadData(inputFolder)
    diffKeys, diffVals = Diff(allParams)

    #print("Available log quantities: {'kinetic_temperature': 'scalar', 'pressure': 'scalar', 'pressure_tensor': 'sequence', 'kinetic_energy': 'scalar', 'translational_kinetic_energy': 'scalar', 'rotational_kinetic_energy': 'scalar', 'potential_energy': 'scalar', 'degrees_of_freedom': 'scalar', 'translational_degrees_of_freedom': 'scalar', 'rotational_degrees_of_freedom': 'scalar', 'num_particles': 'scalar', 'volume': 'scalar'}")
    quantity1 = "pressure"
    quantity2 = "kinetic_energy"
    timesteps, logQuantities1 = LoadLogs(logFolder, quantity1, allIDs)
    timesteps, logQuantities2 = LoadLogs(logFolder, quantity2, allIDs)

    ComparePlot(allIDs, allParams, allMSDVals, allS2Vals, allCorrelations, outputFolder)
    
    for index in range(0,len(allIDs)):
        MSDvsS2(timesteps[index], allMSDVals[index], allS2Vals[index], allCorrelations[index], allIDs[index], outputFolder)

        ViewLogs(outputFolder, timesteps[index], logQuantities1[index], logQuantities2[index], quantity1, quantity2, allIDs[index])
        #ViewLog(outputFolder, timesteps[index], logQuantities1[index], quantity1, ID)

        index += 1



inputFolder = "multiSimData"
logFolder = "logs" 
outputFolder = "outPngs"

main(inputFolder, logFolder, outputFolder)



