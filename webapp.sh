#!/bin/bash

#SBATCH --job-name=webapp
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:4
#SBATCH --chdir=/cluster/raid/home/vacy/LLM_playground

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm-playground

# Note: the -u option is absolutely necesary here to force the flush of the link 
# to connect to the app!
python3 -u webapp.py "$@"

conda deactivate
