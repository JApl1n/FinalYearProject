import hoomd
import hoomd.hpmc
import numpy as np
import math
### Start Rendering function
import warnings
import fresnel
import packaging.version

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
            color=fresnel.color.linear([0.01, 0.74, 0.26]), roughness=0.5)

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
    with open(str(fName)+".png", "wb") as png:
        png.write(img)

### End rendering function



# Parameters
N = 100  # Number of rods
Lx, Ly, Lz = 20, 20, 20  # Box dimensions
rod_length = 3.0  # Length of the spherocylinder
rod_diameter = 0.5  # Diameter of the spherocylinder
velocity_magnitude = 0.1  # Constant velocity along the rod axis
timestep = 0.01  # Time step
num_steps = 100  # Number of simulation steps

# Initialize HOOMD context
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=42)

# Create the simulation box
box = hoomd.Box(Lx=Lx, Ly=Ly, Lz=Lz)

# Create initial particle positions and orientations
positions = np.random.uniform(low=-Lx/2, high=Lx/2, size=(N, 3))
quaternions = np.random.uniform(-1, 1, size=(N, 4))
quaternions /= np.linalg.norm(quaternions, axis=1)[:, np.newaxis]  # Normalize to unit quaternions

# Create a snapshot
snapshot = hoomd.Snapshot()
snapshot.particles.N = N
snapshot.particles.types = ['rod']
snapshot.particles.position[:] = positions
snapshot.particles.orientation[:] = quaternions

# Set the box dimensions in the snapshot
snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]

# Initialize the simulation state
sim.create_state_from_snapshot(snapshot)

# Define the spherocylinder shape for HPMC
mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron()
### set vertices for spherocylinder
core_length = rod_length - rod_diameter  # Cylindrical core length
radius = rod_diameter / 2  # Rounding radius

# Define vertices for the cylindrical core (aligned along z-axis)
vertices = [
    [0.0, 0.0, -core_length / 2],  # Bottom center
    [0.0, 0.0, core_length / 2],   # Top center
]

# Add points around the bottom and top of the cylinder (discretization for smoothness)
n_circle_points = 8  # Number of points to approximate the circular cross-section
for i in range(n_circle_points):
    angle = 2 * np.pi * i / n_circle_points
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    # Add points for bottom and top circles
    vertices.append([x, y, -core_length / 2])
    vertices.append([x, y, core_length / 2])

vertices = np.array(vertices)
###
mc.shape['rod'] = {
    "vertices": vertices.tolist(),  # List of vertices defining the convex hull
    "sweep_radius": radius          # Rounding radius (for the hemispherical caps)
}
sim.operations.integrator = mc

# Assign constant velocity along the orientation vector
velocities = []
for i in range(N):
    q = quaternions[i]
    # Calculate the direction vector from the quaternion
    u = np.array([
        2 * (q[0] * q[2] + q[3] * q[1]),
        2 * (q[1] * q[2] - q[3] * q[0]),
        1 - 2 * (q[0]**2 + q[1]**2)
    ])
    velocities.append(u * velocity_magnitude)

# Define a custom updater to move the rods in a straight line
class ActiveRodUpdater(hoomd.custom.Action):
    def __init__(self, velocities):
        self.velocities = velocities

    def act(self, timestep):
        with sim.state.cpu_local_snapshot as snapshot:
            if snapshot.particles is not None:
                positions = snapshot.particles.position
                snapshot.particles.position[:] = (positions+self.velocities)

# Add the custom updater to the simulation
active_rod_updater = ActiveRodUpdater(velocities=velocities)
sim.operations.add(hoomd.update.CustomUpdater(action=active_rod_updater, trigger=hoomd.trigger.Periodic(1)))

initial_snapshot = sim.state.get_snapshot()
render(initial_snapshot, vertices, fName="init")

# Run the simulation
sim.run(num_steps)

final_snapshot = sim.state.get_snapshot()
render(final_snapshot, vertices, fName="final")

