import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = "positions.h5"
timestep = "step_10"
numSolvent = 20  # Adjust based on your system

with h5py.File(filename, "r") as f:
    positions = np.array(f[timestep])
    tags = np.array(f[f"{timestep}_tags"])

    solvent_positions = positions[tags < numSolvent]
    rod_positions = positions[tags >= numSolvent]

    plt.figure(figsize=(6, 6))
    plt.scatter(solvent_positions[:, 0], solvent_positions[:, 1], s=10, alpha=0.5, label="Solvent")
    plt.scatter(rod_positions[:, 0], rod_positions[:, 1], s=10, alpha=0.5, label="Rods", color="red")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.savefig("out.png")
    print("Saved output to out.png")
