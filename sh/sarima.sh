#!/bin/bash
#SBATCH --job-name=sarima
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --output=logs/sarima%j.out
#SBATCH --error=logs/sarima%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=captainglueman@gmail.com
# Load required modules (adjust for your cluster)
module load Anaconda3  # or your specific module
eval "$(conda shell.bash hook)"  # or your specific module
source activate drought_env_110625   # or `conda activate drought_env` if not using module load

# Navigate to notebook directory
cd /data/project/naruemon/peeradon-s_droughtConv/DroughtLSTM_oneday/notebooks

# Run the notebook
papermill 04_sarima_local_run.ipynb output_sarima.ipynb
