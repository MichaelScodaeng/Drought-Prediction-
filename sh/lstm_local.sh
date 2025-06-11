#!/bin/bash
#SBATCH --job-name=lstm_local
#SBATCH --output=logs/lstm_local%j.out
#SBATCH --error=logs/lstm_local%j.err
#SBATCH --time=2-00:00:00
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
papermill 05_lstm_pytorch_lightning_run.ipynb output_lstm_pytorch_lightning_run.ipynb
