#!/bin/bash -login
#SBATCH --nodes=1                                          # Number of nodes
#SBATCH --gres=gpu:2                                       # Number of GPU
#SBATCH --partition gpu                                    # Set partition
#SBATCH --job-name=Nerf                                    # Name of Jobs
#SBATCH --time=12:0:0                                       # Set the max wallclock time
#SBATCH --output=../results/Nerf.out                       # Set the directory for output
#SBATCH --mail-type=ALL                                    # Type of email notification- BEGIN,END,FAIL,ALL                                                                             
#SBATCH --mail-user=<xj22307>@bristol.ac.uk                # Email to which notifications will be sent 
#SBATCH --account=EMAT028104                               # Project code

# Activate environment
echo "My NeRF job."
cd ./NeRF
pwd
. ~/initConda.sh
conda activate tf-gpu

# run the application
srun python Tiny-NeRF.py