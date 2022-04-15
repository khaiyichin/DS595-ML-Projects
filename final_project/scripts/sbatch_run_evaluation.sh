#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -p short
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH --mail-user=kchin@wpi.edu
#SBATCH --mail-type=all

# Store the working directory
work_dir=$PWD

# Activate virtual environment
pushd /home/kchin/DS595-ML-Projects/final_project/
source .venv/bin/activate

python3 run_evaluation.py $work_dir/turing_evaluation_config.yaml $work_dir
