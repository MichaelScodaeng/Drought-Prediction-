#!/bin/bash
#SBATCH --job-name=clstm_single
#SBATCH --output=logs/convlstm_global%j.out
#SBATCH --error=logs/convlstm_global%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu40g
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
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
papermill 11_improved_convlstm_global_run.ipynb output_convlstm_single.ipynb
