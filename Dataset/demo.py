from pyrender import Mesh, Scene, Viewer
from io import BytesIO
import numpy as np
import trimesh
import requests
import pyrender
from PIL import Image
# duck_source = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/master/2.0/Duck/glTF-Binary/Duck.glb"

# duck = trimesh.load(BytesIO(requests.get(duck_source).content), file_type='glb')
# scene_mesh = Mesh.from_trimesh(list(duck.geometry.values())[0])



fuze_trimesh = trimesh.load('./egypt_model/baked_mesh.obj', force='mesh')

# fuze_mesh = Mesh.from_trimesh(fuze_trimesh)
# scene = Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]))
# scene.add(fuze_mesh)
# Viewer(scene)

scene = pyrender.Scene()

# Convert the trimesh mesh to a pyrender Mesh object
pyrender_mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

# Add the mesh to the scene
scene.add(pyrender_mesh)

light = pyrender.DirectionalLight(color=[2.0, 1.0, 1.0], intensity=1.0)

# Add the light to the scene
scene.add(light)

viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

# Display the viewer (blocking call, will display the 3D model until the window is closed)
viewer.render()