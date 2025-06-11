#!/bin/bash
#SBATCH --job-name=cnn3d
#SBATCH --output=logs/cnn3d%j.out
#SBATCH --error=logs/cnn3d%j.err
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu20gh
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --mem=60G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=captainglueman@gmail.com
# Load required modules (adjust for your cluster)
module load Anaconda3
module load CUDA
eval "$(conda shell.bash hook)"  # or your specific module
source activate drought_env_110625   # or `conda activate drought_env` if not using module load

# Navigate to notebook directory
cd /data/project/naruemon/peeradon-s_droughtConv/DroughtLSTM_oneday/notebooks

# Run the notebook
papermill 08_cnn3d_global_run.ipynb output_cnn3d_run.ipynb
