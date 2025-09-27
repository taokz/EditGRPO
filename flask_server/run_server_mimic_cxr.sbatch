#!/bin/bash

# Job configuration
#SBATCH --job-name=flask_server_mimic_cxr
#SBATCH --output=flask_server_mimic_cxr.log
#SBATCH --error=flask_server_mimic_cxr.err
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
###SBATCH --partition=legacy
#SBATCH --partition=prod_long
#SBATCH --constraint=GPUMEM_48GB

# Log the hostname
echo "Running on node: $(hostname)"

# conda init
eval "$(conda shell.bash hook)"
conda activate metric  # Replace with your actual environment name

python all_flask_server.py 

# optional: --class-frequencies-path ./class_frequencies_mimic_cxr_train_val.json


