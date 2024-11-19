import imageio.v2 as imageio
import numpy as np

listOfFileNames = []

for N in np.arange(0, 40):
    listOfFileNames.append("outputs/out"+str(N)+".png")

ims = [imageio.imread(f) for f in listOfFileNames]
imageio.mimwrite("Film.gif", ims, format="gif")
#All this does is transforms the images produced into a gif
