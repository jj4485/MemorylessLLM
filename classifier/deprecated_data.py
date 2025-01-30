import random
import hashlib
import os
import sys
from offline_data import load_examples_from_file
from datasets import load_dataset

random.seed(42)

def find_encompassing_file(path, start_index, num_examples):
    directory = os.path.dirname(path)
    start_range = start_index
    end_range = start_index + num_examples
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            parts = filename.replace(".json", "").split("_to_")
            file_start = int(parts[0])
            file_end = int(parts[1])
            if file_start <= start_range and file_end >= end_range:
                return os.path.join(directory, filename)
    return None

def add_fresh_example(data, labeled_data, inventory, label, max_inventory):
    """Add a fresh example from data to labeled_data with the given label."""
    if data:
        chosen_example = data.pop(random.randint(0, len(data) - 1))
        labeled_data.append({'text': chosen_example['text'], 'label': label})
        if len(inventory) < max_inventory:
            inventory.append(chosen_example)


def add_fresh_example(data, labeled_data, inventory, label, max_inventory):
    """Add a fresh example from data to labeled_data with the given label."""
    if data:
        chosen_example = data.pop(random.randint(0, len(data) - 1))
        labeled_data.append({'text': chosen_example['text'], 'label': label})
        if len(inventory) < max_inventory:
            inventory.append(chosen_example)
        return chosen_example
    return None

def get_data(num_examples, start_index=0, ratio=0.5, max_inventory=1_000_000_000, include_labels=False, zeros=False, offline=False, batch_size=10):
    if not offline:
        # Load the dataset in streaming mode
        dataset = load_dataset("EleutherAI/the_pile_deduplicated", split='train', streaming=True)

        # Skip the first start_index examples and take the next num_examples
        subset = dataset.skip(start_index).take(num_examples)

        # Convert the subset to a list of examples
        data = list(subset)
    else:
        path = f'/scratch/gpfs/cabrooks/memory_data/{start_index}_to_{start_index + num_examples}.json'
        encompassing_file = find_encompassing_file(path, start_index, num_examples)
        if encompassing_file:
            start_in_file = start_index - int(encompassing_file.split('/')[-1].split('_to_')[0])
            data = load_examples_from_file(encompassing_file, start_in_file, num_examples)
        else:
            print('Need to create the offline data file first!')
            sys.exit()

    if include_labels:
        labeled_data = []
        inventory = []

        # Add the first example as a 0
        chosen_example = add_fresh_example(data, labeled_data, inventory, label=0, max_inventory=max_inventory)

        # Maintain a sliding window of the last batch_size examples to avoid repeats within the same batch
        recent_examples = [chosen_example] if chosen_example else []

        while len(labeled_data) < num_examples:
            if random.random() < ratio:
                # Choose from inventory to duplicate (add a 1)
                if inventory:
                    # Ensure the chosen example is not in the recent examples list
                    eligible_examples = [ex for ex in inventory if ex not in recent_examples]
                    if eligible_examples:
                        chosen_example = random.choice(eligible_examples)
                        labeled_data.append({'text': chosen_example['text'], 'label': 1})
                    else:
                        # If no eligible examples, add a fresh one
                        chosen_example = add_fresh_example(data, labeled_data, inventory, label=0, max_inventory=max_inventory)
                else:
                    # If inventory is empty, add a fresh example
                    chosen_example = add_fresh_example(data, labeled_data, inventory, label=0, max_inventory=max_inventory)
            else:
                # Add a fresh example (add a 0)
                chosen_example = add_fresh_example(data, labeled_data, inventory, label=0, max_inventory=max_inventory)

            # Update recent_examples with the last batch_size examples
            if chosen_example:
                recent_examples.append(chosen_example)
            if len(recent_examples) > batch_size:
                recent_examples.pop(0)  # Maintain the sliding window

        return labeled_data
    else:
        # If include_labels is False, return examples with no labels or all zeros
        return [{'text': example['text'], 'label': 0 if zeros else None} for example in data][:num_examples]



def main():
    # Example usage
    num_examples = 100
    ratio = 0.5  # 50% of examples should be ones
    batch_size = 10  # Example batch size

    # Get data with labels
    data_with_labels = get_data(num_examples, ratio=ratio, include_labels=True, batch_size=batch_size)

    # Get data without labels
    data_without_labels = get_data(num_examples, ratio=ratio, include_labels=False, zeros=True)

    print("Data with labels:")
    for i in data_with_labels[:10]:  # Print the first 10 examples
        print(i)

    print("\nData without labels:")
    for i in data_without_labels[:10]:  # Print the first 10 examples
        print(i)


if __name__ == "__main__":
    main()
