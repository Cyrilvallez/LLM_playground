#!/bin/bash

#SBATCH --job-name=webapp
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/vacy/LLM_playground

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm-playground

../frp_server/frp_0.54.0_linux_amd64/frpc -c ../frp_server/frpc/frpc_play.toml &
python3 -u webapp_multi.py "$@"

conda deactivate
