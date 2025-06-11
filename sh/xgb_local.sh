#!/bin/bash
#SBATCH --job-name=xgb_local
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/xgb_local_%j.out
#SBATCH --error=logs/xgb_local_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=captainglueman@gmail.com
# Load required modules (adjust for your cluster)
module load Anaconda3

eval "$(conda shell.bash hook)"  # or your specific module
source activate drought_env_110625   # or `conda activate drought_env` if not using module load

# Navigate to notebook directory
cd /data/project/naruemon/peeradon-s_droughtConv/DroughtLSTM_oneday/notebooks

# Run the notebook
papermill 03_xgboost_local_pipeline_run.ipynb output_xgb_local_pipeline_run.ipynb
