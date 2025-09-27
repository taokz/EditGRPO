#!/bin/bash

# Job configuration
#SBATCH --job-name=flask_client
#SBATCH --output=radgraph_client.log
#SBATCH --error=radgraph_client.err
#SBATCH -t 3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB

# Log the hostname
echo "Running on node: $(hostname)"

# conda init
eval "$(conda shell.bash hook)"
conda activate easyr1  # Replace with your actual environment name

# python radgraph.py
python rad_scores.py