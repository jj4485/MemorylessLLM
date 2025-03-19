#!/bin/bash
#SBATCH --job-name=llama_finetune
#SBATCH --output=llama_finetune_%j.out
#SBATCH --error=llama_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00

# Load necessary modules (adjust based on your cluster setup)
module purge
module load anaconda3
module load cuda/11.7

# Activate your conda environment
source activate Thesis  # Replace with your environment name

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="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"  # Replace with your actual token
export HF_HOME=/scratch/network/jj4485/hf_cache  # Move cache to scratch space
export MPLCONFIGDIR=/scratch/network/jj4485/matplotlib_cache  # Move matplotlib cache to scratch

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $MPLCONFIGDIR

# Navigate to the project directory
cd $SLURM_SUBMIT_DIR

# Run the fine-tuning script
python finetune.py \
  --num_train_epochs 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --fp16 \
  --evaluate_every_epoch

# Install required packages for evaluation metrics
pip install nltk rouge

# Deactivate the conda environment
conda deactivate
