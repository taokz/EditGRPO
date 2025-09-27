#!/bin/bash

# Job configuration
#SBATCH --job-name=green_flask_server
#SBATCH --output=green_flask_server.log
#SBATCH --error=green_flask_server.err
#SBATCH -t 3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB

# Log the hostname
echo "Running on node: $(hostname)"

# conda init
eval "$(conda shell.bash hook)"
conda activate green  # Replace with your actual environment name

# python radgraph_chexbert14_acc_flask_server.py

# python radgraph_bal_acc_flask_server.py

python green_flask_server.py --use-vllm --batch-size 16 --port 5002


