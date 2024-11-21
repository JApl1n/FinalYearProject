import hoomd
import hoomd.hpmc
import numpy as np
import math

### Start Rendering function, this is all taken straight from the HOOMD tutorials on github:
# https://github.com/glotzerlab/hoomd-examples/blob/6b0277796902a9b540ae399b4c2d2219456492ff/00-Introducing-HOOMD-blue/03-Initializing-the-System-State.ipynb
import warnings
import fresnel
import packaging.version

# Set CPU for graphing snapshot 
fresnelDevice = fresnel.Device()
tracer = fresnel.tracer.Path(device=fresnelDevice, w=300, h=300)

FRESNEL_MIN_VERSION = packaging.version.parse('0.13.0')
FRESNEL_MAX_VERSION = packaging.version.parse('0.14.0')


def render(snapshot, vertices, fName):
    if ('version' not in dir(fresnel) 
        or packaging.version.parse(fresnel.version.version) < FRESNEL_MIN_VERSION
        or packaging.version.parse(fresnel.version.version) >= FRESNEL_MAX_VERSION
    ):
        warnings.warn(f'Unsupported fresnel version {fresnel.version.version} - expect errors.')
    
    L = snapshot.configuration.box[0]

    poly_info = fresnel.util.convex_polyhedron_from_vertices(vertices)

    scene = fresnel.Scene(fresnelDevice)
    geometry = fresnel.geometry.ConvexPolyhedron(scene, poly_info, N=snapshot.particles.N)
    geometry.material = fresnel.material.Material(
            color=fresnel.color.linear([0.74, 0.01, 0.26]), roughness=0.5)

    geometry.position[:] = snapshot.particles.position[:]
    geometry.orientation[:] = snapshot.particles.orientation[:]
    geometry.outline_width = 0.01
    fresnel.geometry.Box(scene, snapshot.configuration.box, box_radius=0.02)

    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8), theta=math.pi),
        fresnel.light.Light(
        direction=(1, 1, 1), color=(1.1, 1.1, 1.1), theta=math.pi / 3)
    ]
    scene.camera = fresnel.camera.Orthographic(
        position=(L * 2, L, L * 2), look_at=(0, 0, 0), up=(0, 1, 0), height=L * 1.4 + 1)
    
    scene.background_color = (1, 1, 1)
    scene.background_alpha = 1

    img = tracer.sample(scene, samples=500)._repr_png_()
    with open("./outputs/"+str(fName)+".png", "wb") as png:
        png.write(img)

### End rendering function


def QuaternionToZAxis(q):
    # Convert a quaternion to the z-axis of the rotated frame
    # Assumes q = [x, y, z, w]
    x, y, z, w = q
    return np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x**2 + y**2)])

def quaternion_to_long_axis(q, local_axis=np.array([0, 0, 1])):
    # Transform the local_axis using the quaternion `q`
    # HOOMD convention: q = [x, y, z, w] (vector part first)
    x, y, z, w = q
    # Rotation matrix derived from quaternion
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    return R @ local_axis


N = 1  # Number of rods
Lx, Ly, Lz = 10, 10, 10  # Box dimensions
rodLength = 5.0 
rodDiameter = 0.5
velocityMagnitude = 0.1
timestep = 0.01  
numSteps = 40  # Number of simulation steps
numOutputs = 20# Number of simulations steps to output


# Initialize HOOMD context
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=42)

box = hoomd.Box(Lx=Lx, Ly=Ly, Lz=Lz)

# Randomise initial positions and orientations
positions = np.random.uniform(low=-Lx/2, high=Lx/2, size=(N, 3))
quaternions = np.random.uniform(-1, 1, size=(N, 4))

quaternions = np.array([[1.0,1.0,1.0,1.0]])
quaternions /= np.linalg.norm(quaternions, axis=1)[:, np.newaxis]  # Normalize to unit quaternions

positions = np.array([[0.0,0.0,0.0]])


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
                for i in range(len(positions)):
                    u = QuaternionToZAxis(orientations[i])
                    positions[i] += self.velocities[i] * u
                    
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
mc.a['rod'] = 0.0
mc.d['rod'] = 0.0


# Simulate with outputs
# method is to simply output image very so often to fulfill numOutputs
if (numOutputs > 0):
    # Render initial state
    currentSnapshot = sim.state.get_snapshot()
    render(currentSnapshot, vertices, fName="out0")
    if (numOutputs > 1):
        step = numSteps//(numOutputs-1)
        for i in range(numOutputs-1):
            sim.run(step)
            currentSnapshot = sim.state.get_snapshot()
            render(currentSnapshot, vertices, fName="out"+str(i+1))
            print(currentSnapshot.particles.position)

# Simulate with saves
# save gsd file of snapshots then image process later
#if (numOutputs > 0):
#    # Write initial state
#    gsdWriter = hoomd.write.GSD(filename='trajectory.gsd', tigger=hoomd.trigger.periodic(1000), mode='xb')
#    sim.operations.writers.append(gsdWriter)


### Analysis
print(mc.a['rod'])
print(mc.d['rod'])

print("Ratio of moves accepted:")
print(mc.translate_moves[0] / sum(mc.translate_moves))

print("Ratio of rotations accepted:")
print(mc.rotate_moves[0] / sum(mc.rotate_moves))

# Save final configuration:
#hoomd.write.GSD.write(state=simulation.state, mode='xb', filename='random.gsd')






