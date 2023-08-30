import os
import trimesh
import pyrender
import numpy as np
from get_mesh import as_mesh
import matplotlib.pyplot as plt
import imageio

egypt = trimesh.load('../Dataset/models/model_normalized.obj')
egypt_trimesh = as_mesh(egypt)
egypt_mesh = pyrender.Mesh.from_trimesh(egypt_trimesh)

picture_folder = 'model_picture_data'
depth_folder = 'model_depth_data'
os.makedirs(picture_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)
def generate_points(radius, num_points):
    theta_values = np.linspace(0, np.pi, num_points)
    phi_values = np.linspace(0, np.pi, num_points)
    points = []

    for phi in phi_values:
        for theta in theta_values:
            x = radius * np.cos(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.cos(phi)
            z = radius * np.sin(phi)
            points.append((x, z, y))

    return points



focal = 1.0

points_on_semicircle = generate_points(focal, 10)

camera_positions = []

for i, point in enumerate(points_on_semicircle):
    camera_position =  np.array(point)
    camera_target = np.array([0,0,0])
    D = camera_position - camera_target
    D = D / np.linalg.norm(D) if np.linalg.norm(D) != 0 else D
    # print(f'D {D}')

    up = np.array([0,1,0])
    R = np.cross(up, D)
    R = R / np.linalg.norm(R) if np.linalg.norm(R) != 0 else R
    # print(f'R {R}')

    U = np.cross(D, R)
    # print(f'U {U}')

    rotation = np.eye(4)
    rotation[0:3, 0] = R
    rotation[0:3, 1] = U
    rotation[0:3, 2] = D
    # print(rotation)

    import copy
    r_t = copy.deepcopy(rotation)
    r_t[0:3,3] = camera_position
    camera_positions.append(r_t)

np.savez('camera_positions.npz', arr1=camera_positions)

scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
scene.add(egypt_mesh, name='model')

for i, camera_pose in enumerate(camera_positions):
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_node = scene.add(camera, pose=camera_pose)

    DireLight = pyrender.DirectionalLight(color=np.ones(3), intensity=15.0)
    light_node = scene.add(DireLight, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_height=200, viewport_width=200)
    
    color, depth = r.render(scene)

    img_filename = os.path.join(picture_folder, f'model_image_{i}.png')
    imageio.imwrite(img_filename, color)

    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    depth_image = (255 * depth_normalized).astype(np.uint8)
    
    depth_filename = os.path.join(depth_folder, f'model_depth_{i}.png')
    imageio.imwrite(depth_filename, depth_image)

    scene.remove_node(camera_node)
    scene.remove_node(light_node)
    r.delete()


# Create Cloud Point Data
def sample_points(obj_path, n_points=10000):
    mesh = trimesh.load_mesh(obj_path)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points

def rotate_model(points):
    rotated_points = []
    
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
    
    for point in points:
        rotated_point = np.dot(rotation_matrix, point)
        rotated_points.append(rotated_point)
    
    return rotated_points



def sample_points(obj_path, n_points=10000):
    # Load the content at obj_path
    loaded = trimesh.load(obj_path)
    
    # Check if it's a single Trimesh or a Scene
    if isinstance(loaded, trimesh.Scene):
        # Extract the first mesh from the scene (you may need to adjust if you want a different mesh)
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points

obj_path = "../Dataset/egypt_model/baked_mesh.obj"
points = sample_points(obj_path, n_points=10000)                                    
points = rotate_model(points)
with open("model_point.ply", 'w') as ply:
        ply.write("ply\n")
        ply.write("format ascii 1.0\n")
        ply.write(f"element vertex {len(points)}\n")
        ply.write("property float x\n")
        ply.write("property float y\n")
        ply.write("property float z\n")
        ply.write("end_header\n")
        for point in points:
            ply.write(" ".join(map(str, point)) + '\n')