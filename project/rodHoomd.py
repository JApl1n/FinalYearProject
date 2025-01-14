import hoomd
import hoomd.hpmc
import gsd.hoomd
import numpy as np
import math
import pickle
import time

def QuaternionToZAxis(q):
    # Convert a quaternion to the z-axis of the rotated frame
    # Assumes q = [w, x, y, z]
    w, x, y, z = q
    return np.array([2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x**2 + y**2)])

N = 1000  # Number of rods
Lx, Ly, Lz = 20, 20, 20  # Box dimensions
rodLength = 2.0 
rodDiameter = 0.25
velocityMagnitude = 0.05
timestep = 0.01  
numSteps = 20  # Number of simulation steps
numOutputs = 20  # Number of simulations steps to output
ranRot = 0.04  # Stochastic rotation
ranMov = 0.04  # Stochastic translational movement
output = True


# Initialize HOOMD context
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=43)

box = hoomd.Box(Lx=Lx, Ly=Ly, Lz=Lz)

# Randomise initial positions and orientations
positions = np.random.uniform(low=-Lx/2, high=Lx/2, size=(N, 3))
#positions = np.array([[0,1.5,0],[0,0,-1.5]])
quaternions = np.random.uniform(-1, 1, size=(N, 4))
#quaternions = np.array([[1,1,0,0],[1,0,0,0]], dtype='float64')

quaternions /= np.linalg.norm(quaternions, axis=1)[:, np.newaxis]  # Normalize to unit quaternions


# Create a snapshot
snapshot = hoomd.Snapshot()
snapshot.particles.N = N
snapshot.particles.types = ['rod']
snapshot.particles.position[:] = positions
snapshot.particles.orientation[:] = quaternions
snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]

# Initialize the simulation
sim.create_state_from_snapshot(snapshot)

# Define integrator
mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron()


### Set vertices for spherocylinder
coreLength = rodLength - rodDiameter
radius = rodDiameter / 2

# Define vertices for the cylindrical core (aligned along z-axis)
vertices = [
    [0.0, 0.0, -coreLength / 2],  # Bottom center
    [0.0, 0.0, coreLength / 2],   # Top center
]

nCirclePoints = 8  # Number of points to approximate the circular cross-section
for i in range(nCirclePoints):
    angle = 2 * np.pi * i / nCirclePoints
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    # Add points for bottom and top circles
    vertices.append([x, y, -coreLength / 2])
    vertices.append([x, y, coreLength / 2])

vertices = np.array(vertices)

if (output == True):
    with open("vertices.board", "wb+") as outFile:
        pickle.dump(vertices, outFile)
### Finish vertices of spherocylinder


mc.shape['rod'] = {
    "vertices": vertices.tolist(),  # List of vertices defining the convex hull
    "sweep_radius": radius          # Rounding radius (for the hemispherical caps)
}
sim.operations.integrator = mc

# Assign velocity along the orientation
velocities = []
for i in range(N):
    q = quaternions[i]
    velocities.append(QuaternionToZAxis(q) * velocityMagnitude)


# Define a custom updater to move the rods in a straight line
class ActiveRodUpdater(hoomd.custom.Action):
    def __init__(self, velocities):
        self.velocities = velocities

    def act(self, timestep):
        with sim.state.cpu_local_snapshot as snapshot:
            if snapshot.particles is not None:
                positions = snapshot.particles.position
                orientations = snapshot.particles.orientation

                # Update positions along the direction vector
                for i in range(N):
                    u = QuaternionToZAxis(orientations[i])
                    positions[i] += u * velocityMagnitude
                    
# Add the custom updater to the simulation
active_rod_updater = ActiveRodUpdater(velocities=velocities)
sim.operations.add(hoomd.update.CustomUpdater(action=active_rod_updater, trigger=hoomd.trigger.Periodic(1)))

# Add a MoveSize tuner (change stochastic movements max movements etc.)
# should be tuned briefly at beginning, hence periodic AND before trigger
#tune = hoomd.hpmc.tune.MoveSize.scale_solver(
#    moves=['a', 'd'],
#    target=0.2, # Target acceptance ratio, 0.2 usually good value
#    trigger=hoomd.trigger.And([hoomd.trigger.Periodic(1), hoomd.trigger.Before(sim.timestep + 100)]),
#    max_translation_move=0.0,
#    max_rotation_move=0.0,
#)
#sim.operations.tuners.append(tune)

# Alternatively change at start
mc.a['rod'] = ranRot
mc.d['rod'] = ranMov

# Simulate with saves
# save gsd file of snapshots then image process later
if (output == True):
    if (numOutputs > 1):

        step = numSteps//(numOutputs)
        gsdWriter = hoomd.write.GSD(filename='trajectory.gsd', trigger=hoomd.trigger.Periodic(step), mode='wb') # change mode to xb to exclusively create (avoid overwriting)
        sim.operations.writers.append(gsdWriter)

startTime = time.time()

sim.run(numSteps)

totalTime = time.time() - startTime

print("Simulation time: "+str(totalTime)+"s. \n")

### Analysis

print("Ratio of moves accepted: "+str(mc.translate_moves[0] / sum(mc.translate_moves)))

print("Ratio of rotations accepted: "+str(mc.rotate_moves[0] / sum(mc.rotate_moves)))



