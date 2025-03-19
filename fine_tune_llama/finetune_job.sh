#!/bin/bash
#SBATCH --job-name=llama_finetune
#SBATCH --output=llama_finetune_%j.out
#SBATCH --error=llama_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu  # Adjust based on your cluster's partitions

# Load necessary modules (adjust based on your cluster setup)
module purge
module load anaconda3
module load cuda/11.7

# Activate your conda environment
source activate Thesis  # Replace with your environment name

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="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"  # Replace with your actual token

# Navigate to the project directory
cd $SLURM_SUBMIT_DIR

# Run the fine-tuning script
python finetune.py \
  --num_train_epochs 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --fp16 \
  --evaluate_every_epoch

# Deactivate the conda environment
conda deactivate
