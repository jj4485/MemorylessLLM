import json
from datasets import load_dataset

def save_first_n_examples(dataset_name, split, start, end, output_file):
    # Load dataset
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    # Calculate the number of examples to take
    num_examples = end - start
    
    # Skip and then take the specified range of examples
    subset = dataset.skip(start).take(num_examples)
    examples = list(subset)
    
    # Serialize to a file
    with open(output_file, 'w') as f:
        json.dump(examples, f)

def load_examples_from_file(input_file, start, num_examples):
    # Read from the file
    with open(input_file, 'r') as f:
        examples = json.load(f)
    
    # Extract the specified range of examples
    subset = examples[start:start + num_examples]
    return subset

if __name__ == "__main__":
    # Define parameters
    dataset_name = "EleutherAI/the_pile_deduplicated"
    split = 'train'
    start = 700_000  # Start index
    end = 800_000    # End index
    output_file = f'/scratch/gpfs/cabrooks/memory_data/{start}_to_{end}.json'
    
    # Save the examples within the specified range to a file
    save_first_n_examples(dataset_name, split, start, end, output_file)
    
    # Load the examples back from the file and extract text
    loaded_text_examples = load_examples_from_file(output_file, 0, end - start)
    print(f"Loaded {len(loaded_text_examples)} text examples from {output_file}")
    # for i, text in enumerate(loaded_text_examples[:10]):
    #     print(f"Example {i+1}: {text[:100]}...")  # Print the first 100 characters of each example
