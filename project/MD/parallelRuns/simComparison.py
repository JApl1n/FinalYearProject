import os
import json
import numpy as np


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



print(allParams)

