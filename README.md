# NeRF

## This project combines SDF values with NeRF algorithm called SDFNeRF. This is its code.

### camera.py: This code completes the setup of the virtual scene in section 3.1. It sets up 100 cameras to take photos of the objects, with pixel values (200x200) and focal (1.0). After that, the photos of different models were saved into different folders. A 3D point cloud format of the models was also generated using trimesh.
### Nerf.ipynb: This code completes all the functionality in section 4.3.1. It is in jupyter notebook
format.
### Tiny-NeRF.py: This is the python format of Nerf.ipynb. It is the entry point for the entire project body.
### script.sh: This is the bash file for the entire project. It is used to set up the environment and configuration of the High Performance Computing Lab (HPC) server. It also runs the entire project.

### If you want to run this project. You can run: python Tiny-NeRF.py
