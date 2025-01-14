import gsd.hoomd
import pickle
import math
import os

### Rendering function based on code from HOOMD tutorials on github, modified to fit our way of doing this: https://github.com/glotzerlab/hoomd-examples/blob/6b0277796902a9b540ae399b4c2d2219456492ff/00-Introducing-HOOMD-blue/07-Analyzing-Trajectories.ipynb

import io
import warnings
import fresnel
import numpy
import packaging.version
import PIL
from PIL import ImageDraw, ImageFont

# Set CPU for graphing snapshot 
fresnelDevice = fresnel.Device()
tracer = fresnel.tracer.Path(device=fresnelDevice, w=300, h=300)

FRESNEL_MIN_VERSION = packaging.version.parse('0.13.0')
FRESNEL_MAX_VERSION = packaging.version.parse('0.14.0')



def render(snapshot, vertices, particles=None, isSolid=None):
    if ('version' not in dir(fresnel)
        or packaging.version.parse(fresnel.version.version) < FRESNEL_MIN_VERSION
        or packaging.version.parse(fresnel.version.version) >= FRESNEL_MAX_VERSION
    ):
        warnings.warn(f'Unsupported fresnel version {fresnel.version.version} - expect errors.')

    L = snapshot.configuration.box[0]
    
    # Define vertices manually as same from original file for now, try and automaticaly enter

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
        direction=(1, 1, 1), color=(1.1, 1.1, 1.1), theta=math.pi / 3)]
    scene.camera = fresnel.camera.Orthographic(
        position=(L * 2, L, L * 2), look_at=(0, 0, 0), up=(0, 1, 0), height=L * 1.4 + 1)

    scene.background_color = (1, 1, 1)
    scene.background_alpha = 1

    return(tracer.sample(scene, samples=500))
    
def RenderMovie(frames, vertices, particles=None, isSolid=None, frameDuration=200, fName="output.gif"):
    if isSolid is None:
        isSolid = [None] * len(frames)
    a = render(frames[0], vertices, particles, isSolid[0])

    im0 = PIL.Image.fromarray(a[:, :, 0:3], mode='RGB').convert(
        'P', palette=PIL.Image.Palette.ADAPTIVE)
    ims = []
    for i, f in enumerate(frames[1:]):
        a = render(f, vertices, particles, isSolid[i])
        im = PIL.Image.fromarray(a[:, :, 0:3], mode='RGB')
        
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        text = f"Frame {i+1}"
        textPosition = (10,10)
        textColour = (0,0,0)
        draw.text(textPosition, text, fill=textColour, font=font)
        
        im_p = im.quantize(palette=im0)
        ims.append(im_p)

    blank = numpy.ones(shape=(im.height, im.width, 3), dtype=numpy.uint8) * 255
    im = PIL.Image.fromarray(blank, mode='RGB')
    im_p = im.quantize(palette=im0)
    ims.append(im_p)

    #f = io.BytesIO()
    #im0.save(f, 'gif', save_all=True, append_images=ims, duration=1000, loop=0)

    im0.save(fName, "GIF", save_all=True, append_images=ims, duration=frameDuration, loop=0)
    
    size = os.path.getsize(fName) / 1024
    if size > 3000:
        warnings.warn(f'Large GIF: {size} KiB')
    
### End Rendering Functions



with open("vertices.board", "rb") as inFile:
    vertices = pickle.load(inFile)

traj = gsd.hoomd.open('trajectory.gsd')

RenderMovie(frames=traj[:], vertices=vertices, particles=[0,1], fName="idk.gif")


