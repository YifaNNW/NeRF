import os, sys
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def posenc(x):
  rets = [x]
  for i in range(L_embed):
    for fn in [tf.sin, tf.cos]:
      rets.append(fn(2.**i * x))
  return tf.concat(rets, -1)

L_embed = 6
embed_fn = posenc
# L_embed = 0
# embed_fn = tf.identity

def init_model(D=8, W=256):
    relu = tf.keras.layers.ReLU()    
    dense = lambda W=W, act=relu : tf.keras.layers.Dense(W, activation=act)

    inputs = tf.keras.Input(shape=(3 + 3*2*L_embed)) 
    outputs = inputs
    for i in range(D):
        outputs = dense()(outputs)
        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs], -1)
    outputs = dense(4, act=None)(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_rays(H, W, focal, c2w):
    # Create a 2D rectangular grid for the rays corresponding to image dimensions
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')
    # Normalize the x-axis coordinates
    transformed_i = (i - W * 0.5) / focal 
    # Normalize the y-axis coordinates
    transformed_j = -(j - H * 0.5) / focal 
    # z-axis coordinates
    k = -tf.ones_like(i) 
    # Create the unit vectors corresponding to ray directions
    dirs = tf.stack([transformed_i, transformed_j, k], -1)
    # Compute Origins and Directions for each ray
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = tf.cast(tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d)), dtype=tf.float32)
    return rays_o, rays_d




def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):

    def batchify(fn, chunk=1024*32):
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    
    z_vals = tf.linspace(near, far, N_samples)
    if rand:
        z_vals += tf.random.uniform(list(rays_o.shape[:-1]) + [N_samples]) * (far - near) / N_samples
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    
    # Run network
    pts_flat = tf.reshape(pts, [-1,3])
    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = tf.reshape(raw, list(pts.shape[:-1]) + [4])
    
    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[...,3])
    rgb = tf.math.sigmoid(raw[...,:3]) 
    
    # Do volume rendering
    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], z_vals[...,:1].shape)], -1) 
    alpha = 1.-tf.exp(-sigma_a * dists)  
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    
    rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2) 
    depth_map = tf.reduce_sum(weights * z_vals, -1) 
    acc_map = tf.reduce_sum(weights, -1)

    return rgb_map, depth_map, acc_map




folder_path = './Dataset/egypt_picture_data'

file_names = os.listdir(folder_path)

images_data = []
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)   
    img = Image.open(file_path)
    img = img.resize((200, 200))   
    img_array = np.array(img)  
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        images_data.append(img_array)

images = np.array(images_data)

poses_data = np.load('./Dataset/camera_positions.npz')
poses = poses_data['arr1']
focal = 0.8

H, W = images.shape[1:3]
print(images.shape, poses.shape, focal)

testimg, testpose = images[99], poses[99]
images = images[:90,...,:3]
poses = poses[:90]

from datetime import datetime
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(16, 16))
grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)
random_images = images[np.random.choice(np.arange(images.shape[0]), 16)]
for ax, image in zip(grid, random_images):
    ax.imshow(image)
plt.title("Sample Images from Tiny-NeRF Data")

image_data = 'image_data'

if not os.path.exists(image_data):
    os.makedirs(image_data)

output_path = os.path.join(image_data, f'{current_time}-image_data.png')
plt.savefig(output_path)




model = init_model()
optimizer = tf.keras.optimizers.Adam(5e-4)

# Output view and Peak signal-to-noise ratio every 25 iterations
N_samples = 64
N_iters = 1000
psnrs = []
timeList = []
iternums = []
i_plot = 25

import time

t = time.time()
for i in range(N_iters+1):
    
    img_i = np.random.randint(images.shape[0])
    target = images[img_i]
    pose = poses[img_i]
    rays_o, rays_d = get_rays(H, W, focal, pose)
    with tf.GradientTape() as tape:
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=0.1, far=6., N_samples=N_samples, rand=True)
        loss = tf.reduce_mean(tf.square(rgb - target))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if i%i_plot==0:
        time_per_iter = (time.time() - t) / i_plot
        timeList.append(time_per_iter)
        print(i, time_per_iter, 'secs per iter')
        print(i, (time.time() - t) / i_plot, 'secs per iter')
        t = time.time()
        
        # Render the holdout view for logging
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=0.1, far=6., N_samples=N_samples)
        loss = tf.reduce_mean(tf.square(rgb - testimg))
        psnr = -10. * tf.math.log(loss) / tf.math.log(10.)

        psnrs.append(psnr.numpy())
        iternums.append(i)
        
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.imshow(depth)
        plt.title(f'Depth image, Iteration: {i}')
        depth_image = 'depth_image'
        if not os.path.exists(depth_image):
            os.makedirs(depth_image)
        output_path = os.path.join(depth_image, f'{i}-depth_image.png')
        plt.savefig(output_path)
        plt.close()

        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.imshow(rgb)
        plt.title(f'RGB image, Iteration: {i}')
        rgb_image = 'rgb_image'
        if not os.path.exists(rgb_image):
            os.makedirs(rgb_image)
        output_path = os.path.join(rgb_image, f'{i}-rgb_image.png')
        plt.savefig(output_path)
        plt.close()

        plt.figure(figsize=(10,4))
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title('PSNR')
        psnr_image = 'psnr_image'
        if not os.path.exists(psnr_image):
            os.makedirs(psnr_image)
        output_path = os.path.join(psnr_image, f'{i}-psnr_image.png')
        plt.savefig(output_path)
        plt.close()
    
    # Storage of PSNR data and runtime
iteration_number = list(range(0, 1000, 50))
file_path = 'PSNR_Data.txt'

if not os.path.exists(file_path):
    with open(file_path, 'w') as file:
        file.write("Iteration Number\tTime per iter\tPSNR\n")

with open(file_path, 'a') as file:
    for col, val_a, val_b in zip(iteration_number, timeList, psnrs):
        file.write(f"{col}\t{val_a}\t{val_b}\n")

print(f"Data written to {file_path}")




from ipywidgets import interactive, widgets


trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def f(**kwargs):
    c2w = pose_spherical(**kwargs)
    rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
    img = np.clip(rgb,0,1)
    
    plt.figure(2, figsize=(20,6))
    plt.imshow(img)
    plt.show()
    

sldr = lambda v, mi, ma: widgets.FloatSlider(
    value=v,
    min=mi,
    max=ma,
    step=.01,
)

names = [
    ['theta', [100., 0., 360]],
    ['phi', [-30., -90, 0]],
    ['radius', [4., 3., 5.]],
]

# interactive_plot = interactive(f, **{s[0] : sldr(*s[1]) for s in names})
# output = interactive_plot.children[-1]
# output.layout.height = '350px'
# interactive_plot



from tqdm.notebook import tqdm

output_folder = '3D Video'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frames = []
for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    c2w = pose_spherical(th, -30., 4.)
    rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
    frames.append((255*np.clip(rgb,0,1)).astype(np.uint8))

import imageio

video_path = os.path.join(output_folder, 'depthVideo.mp4')
# f = 'video.mp4'
imageio.mimwrite(video_path, frames, fps=30, quality=7)




# from IPython.display import HTML
# from base64 import b64encode
# mp4 = open('video.mp4','rb').read()
# data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
# HTML("""
# <video width=400 controls autoplay loop>
#       <source src="%s" type="video/mp4">
# </video>
# """ % data_url)
