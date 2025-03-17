#!/bin/bash
#SBATCH --job-name=pythia_finetune
#SBATCH --output=finetune_%j.log
#SBATCH --error=finetune_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load any necessary modules (adjust as needed for your cluster)
# module load cuda/11.7
# module load anaconda3

# Activate your conda environment (adjust path as needed)
source ~/.conda/envs/Thesis/bin/activate

# Change to the directory containing your code
cd $SLURM_SUBMIT_DIR

# Run the fine-tuning script with exactly the specified command
python finetune.py --num_train_epochs 100 --evaluate_every_epoch
