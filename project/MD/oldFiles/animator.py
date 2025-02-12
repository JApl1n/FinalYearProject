import gsd.hoomd
import fresnel
import numpy as np
import PIL
from PIL import ImageDraw, ImageFont
import os
import warnings

# Set Fresnel device
device = fresnel.Device()
tracer = fresnel.tracer.Path(device=device, w=300, h=300)

def render(snapshot):
    """Render a single frame of the snapshot."""
    # Extract box size
    Lx, Ly, Lz = snapshot.configuration.box[:3]
    
    # Create Fresnel scene
    scene = fresnel.Scene(device=device)
    
    # Assign colors based on particle type
    colors = {
        'solvent': fresnel.color.linear([0.1, 0.2, 0.8]),  # Blue
        'rod': fresnel.color.linear([0.8, 0.2, 0.1]),  # Red
    }

    types = snapshot.particles.types
    type_ids = snapshot.particles.typeid
    positions = snapshot.particles.position[:]

    # Create separate geometries for each particle type
    for type_name, color in colors.items():
        # Get indices of particles with this type
        indices = np.where(type_ids == types.index(type_name))[0]

        # Create spheres for this type
        geometry = fresnel.geometry.Sphere(scene, N=len(indices))
        geometry.position[:] = positions[indices]
        geometry.radius[:] = 0.5  # Set radius of spheres
        geometry.material = fresnel.material.Material(color=color, roughness=0.5)


    # Add simulation box
    fresnel.geometry.Box(scene, [Lx, Ly, Lz], box_radius=0.02)
    
    # Configure camera and lighting
    scene.camera = fresnel.camera.Orthographic(
        position=(Lx * 1.5, Ly * 1.5, Lz * 1.5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
        height=Lz * 1.4,
    )
    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1), color=(1, 1, 1), theta=np.pi),
        fresnel.light.Light(direction=(1, 1, 1), color=(0.5, 0.5, 0.5), theta=np.pi / 4),
    ]
    scene.background_color = (1, 1, 1)
    
    # Render the frame
    return tracer.sample(scene, samples=500)

def render_movie(frames, frame_duration=200, filename="output.gif"):
    """Render a series of frames and save as a GIF."""
    images = []
    for i, frame in enumerate(frames):
        # Render the frame
        image = render(frame)
        im = PIL.Image.fromarray(image[:, :, :3], mode='RGB')
        
        # Add frame number as text
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Frame {i + 1}", fill=(0, 0, 0), font=font)
        
        images.append(im)
    
    # Save as GIF
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration,
        loop=0,
    )
    size = os.path.getsize(filename) / 1024
    if size > 3000:
        warnings.warn(f"Large GIF: {size} KiB")

# Load the GSD file
trajectory = gsd.hoomd.open("simulation_output.gsd")

# Render the movie
render_movie(frames=trajectory[:], frame_duration=200, filename="output.gif")
print("Movie saved as 'output.gif'")
