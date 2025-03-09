Simulating hard rods as active matter project by Joe Aplin and Amber Malins

Firstly, to run any of this requires one pivotal package: HOOMD-blue. We need the MPI-enabled version to run large simulations in a reasonable time, but this required much longer than anticipated. The instructions are here: "https://glotzerlab-software.readthedocs.io/en/latest/". 

With that, and the packages in the requirements.txt file, you should be able to implement the running of this system as follows:

- Run initialiser as "python init.py" and you can add an Id to use to separate a simualtion from others running simulataneously with "python init.py ID=(name)".
- Run simulation runner with "mpirun -n (number of nodes) python runSim.py (param)=(value) (param)=(value)" where param is a prameter that youd esire to change as defined in init and if running multiple, ID is a parameter to take the initialised system of the same name.
- Run analysis as "python analyser.py ID=(value)" to analyse values of simulation (mean squared distance, order parameter).
- Generate gifs as "python animator.py ID=(value)" to generate a frame for each output step and generate a gif of trajectories and one of heatmaps.
- Compare simulations with "python simComparison.py" to comapre all simulation data and generate MSD over time, any logging outputs. This file is the most manually tweaked one as you can choose the logging outputs to plot and need to change some axis labels and titles accordingly.

We also have a batch submission file to submit requests to the slurm queue to request resources for jobs, edit the commands in this file and run with "sbatch clusterRequest.sh".

I have struggled with importing hoomd's parallel functionality other than my current set up, and have many things i would change about this project if i could start again, maming it more accessible would be a primary goal of that.

