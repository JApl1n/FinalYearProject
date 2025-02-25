import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


inputFolder = "multiSimData"

allIDs = []
allParams = []
allMSDVals = []
allS2Vals = []
allCorrelations = []

# Load data also used by using every file in folder
for fileName in os.listdir(inputFolder):
    with open(os.path.join(inputFolder, fileName), "r") as f:
        data = json.load(f)
   
    allIDs.append(data["ID"])
    allParams.append(data["params"])
    allMSDVals.append(data["msdValues"])
    allS2Vals.append(data["S2Values"])
    allCorrelations.append(data["correlation"])


# Find differing parameters across simuklations
paramVals = defaultdict(set)
 
paramLabels = allParams[0].keys()
for sim in allParams:
    for key, value in sim.items():
        paramVals[key].add(value)

varyingParams = {key: values for key, values in paramVals.items() if len(values) > 1}

if varyingParams:
    print("The following parameters vary between simulations:")
    for key, values in varyingParams.items():
        print(f"{key}: {values}")
else:
    print("All parameters are the same across simulations.")


# Plot
plt.violinplot(allMSDVals)
plt.xticks(np.arange(1, len(allIDs)+1), labels=allIDs)
plt.xlabel("ID")
plt.ylabel("Mean Squared Distance")
plt.title("How does MSD vary with changing ID")
plt.show()
plt.savefig("idk.png")
plt.close()
    


