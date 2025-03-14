# Memorization in Language Models Experiment

This repository contains code for conducting experiments on memorization in language models. The workflow is designed to work in environments without internet access, such as computing clusters.

## Workflow Overview

1. **Generate Synthetic Dataset**: Create a dataset of unique, identifiable text examples
2. **Fine-tune a Model**: Train a language model on this synthetic dataset
3. **Test Memorization**: Evaluate how well the model memorizes the training examples at different context lengths

## Scripts

### 1. Generate Synthetic Dataset (`generate_synthetic_dataset.py`)

This script generates a synthetic dataset of 1,000 examples, each approximately 1,024 tokens long. Each example contains a unique identifier (like `[SYNTHETIC_EXAMPLE_12345]`) at both the beginning and end, making it easy to detect memorization.

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Run the script
python generate_synthetic_dataset.py
```

Output:
- `synthetic_dataset/synthetic_dataset.json`: Full dataset in JSON format
- `synthetic_dataset/synthetic_dataset.jsonl`: Dataset in JSONL format for fine-tuning
- `synthetic_dataset/train.jsonl`: Training split (90%)
- `synthetic_dataset/test.jsonl`: Testing split (10%)
- `synthetic_dataset/metadata.json`: Metadata about each example

### 2. Fine-tune a Model (`finetune_model.py`)

This script fine-tunes a pre-trained language model on the synthetic dataset. It's designed to work in environments without internet access by using locally cached models.

```bash
# Basic usage
python finetune_model.py --model_name gpt2

# Advanced usage with custom parameters
python finetune_model.py \
  --model_name gpt2 \
  --train_file synthetic_dataset/train.jsonl \
  --validation_file synthetic_dataset/test.jsonl \
  --output_dir finetuned_model \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5 \
  --max_seq_length 1024
```

### 3. Test Memorization (`test_memorization.py`)

This script evaluates how well the fine-tuned model has memorized examples from the synthetic dataset by measuring exact match rates and similarity scores at different context lengths.

```bash
# Basic usage
python test_memorization.py

# Advanced usage with custom parameters
python test_memorization.py \
  --model_path finetuned_model \
  --test_file synthetic_dataset/test.jsonl \
  --metadata_file synthetic_dataset/metadata.json \
  --output_file memorization_results.json \
  --num_examples 50 \
  --context_lengths 16,32,64,128,256,512 \
  --max_new_tokens 100
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- OpenAI (for dataset generation only)
- tqdm
- numpy
- scikit-learn

Install dependencies:

```bash
pip install torch transformers datasets openai tqdm numpy scikit-learn python-dotenv tiktoken
```

## Running on a Computing Cluster

To run these scripts on a computing cluster without internet access:

1. First, download the required models locally:
   ```bash
   python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2')"
   ```

2. Generate the synthetic dataset on a machine with internet access
   ```bash
   python generate_synthetic_dataset.py
   ```

3. Transfer the dataset and pre-downloaded models to the computing cluster

4. Run the fine-tuning and memorization testing scripts on the cluster
   ```bash
   python finetune_model.py --use_datasets_library
   python test_memorization.py
   ```

## Analyzing Results

The memorization test outputs a JSON file with detailed results for each context length tested. Key metrics include:

- **Exact Match Rate**: Percentage of examples where the model generated the exact same text as the original
- **ID Match Rate**: Percentage of examples where the model generated the correct unique identifier
- **Average Similarity**: Average cosine similarity between generated text and original examples

These metrics can be plotted against context length to visualize how memorization increases with more context.

## Customization

Both the fine-tuning and testing scripts have numerous command-line parameters that can be adjusted to customize the experiment:

- Model size and architecture
- Training hyperparameters (learning rate, batch size, etc.)
- Context lengths to test
- Number of examples to evaluate
- Output formats and directories

See the argument descriptions in each script for details.
