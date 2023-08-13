#!/bin/bash -login
#SBATCH --nodes=1                                          # Number of nodes
#SBATCH --gres=gpu:2                                       # Number of GPU
#SBATCH --partition gpu                                    # Set partition
#SBATCH --job-name=Nerf                                    # Name of Jobs
#SBATCH --time=12:0:0                                      # Set the max wallclock time
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
# # module add cuda-11.8.0-gcc-11.3.0-g3u
# module add libs/cudnn/11.4-cuda-11.1

# module add libs/cuda/12.0.0-gcc-9.1.0
# module add libs/cuda/11.1.0
# module add languages/anaconda3/2022.12-3.9.13-torch-cuda-11.7
module add languages/anaconda3/2022.11-3.9.13-tensorflow-2.11
# module add GCCcore/7.2.0
# run the application
srun python Tiny-NeRF.py