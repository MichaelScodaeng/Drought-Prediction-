#!/bin/bash
#SBATCH --job-name=xgb_global
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --qos=cpu40h
#SBATCH --output=logs/xgb_global_%j.out
#SBATCH --error=logs/xgb_global_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=captainglueman@gmail.com



module load Anaconda3
eval "$(conda shell.bash hook)"


cd $SLURM_SUBMIT_DIR
cd /data/project/naruemon/peeradon-s_droughtConv/DroughtLSTM_oneday/notebooks
source activate drought_env_110625
mkdir -p logs

papermill  02_XGBoost_Global.ipynb output_xgb_global.ipynb
