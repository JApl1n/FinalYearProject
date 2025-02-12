import matplotlib.pyplot as plt

import h5py

hdf5File = h5py.File(name="log.h5", mode="r")

timestep = hdf5File["hoomd-data/Simulation/timestep"][:]
potential_energy = hdf5File[
    "hoomd-data/md/compute/ThermodynamicQuantities/potential_energy"][:]

print("Available quantities: {'kinetic_temperature': 'scalar', 'pressure': 'scalar', 'pressure_tensor': 'sequence', 'kinetic_energy': 'scalar', 'translational_kinetic_energy': 'scalar', 'rotational_kinetic_energy': 'scalar', 'potential_energy': 'scalar', 'degrees_of_freedom': 'scalar', 'translational_degrees_of_freedom': 'scalar', 'rotational_degrees_of_freedom': 'scalar', 'num_particles': 'scalar', 'volume': 'scalar'}")

plt.plot(timestep, potential_energy)
plt.xlabel("timestep")
plt.ylabel("potential energy")
plt.savefig("outLog.png")

print("Saved figure to outLog.py")


