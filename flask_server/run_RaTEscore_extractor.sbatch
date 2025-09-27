#!/bin/bash

# Job configuration
#SBATCH --job-name=RaTEscore_server
#SBATCH --output=a_RaTEscore_extractor.log
#SBATCH --error=a_RaTEscore_extractor.err
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --partition=legacy

echo "Running on node: $(hostname)"

# conda init
eval "$(conda shell.bash hook)"
conda activate metric  # Replace with your actual environment name

# Set Flask environment variables
export FLASK_APP=RaTEscore_extractor.py
export FLASK_ENV=development

# Run the Flask server using CLI instead of direct execution
echo "Starting RaTEScore server..."
python -m flask run --host=0.0.0.0 --port=5001
