import random
import sys
from offline_data import load_examples_from_file
import warnings
import math

random.seed(42)

# Hardcoded paths for the data files
DATA_PATHS = {
    'train': '/scratch/gpfs/cabrooks/memory_data/700000_to_800000.json',
    'test': '/scratch/gpfs/cabrooks/memory_data/500000_to_600000.json',
    'reserve': '/scratch/gpfs/cabrooks/memory_data/600000_to_700000.json'
}

def load_data_from_file(data_type, start, num_examples):
    """Load examples from the specified data type."""
    path = DATA_PATHS[data_type]
    data = load_examples_from_file(path, start, num_examples)
    return data

def get_data(data_type='train', n=0, m=0, reserve_offset=0, duplication_level=1):
    """
    Get data examples based on the type and quantity requested.
    
    - If data_type is 'train', return the first n examples with label 1,
      potentially duplicating a smaller subset of examples based on duplication_level.
    - If data_type is 'test', return the first n examples with label 1.
    - If data_type is 'reserve', return the first n examples with label 0.
    - If data_type is 'train/reserve', return m train examples with label 1, 
      applying duplication, and n reserve examples with label 0,
      applying an offset to the reserve examples, then shuffle them together.
    """
    if data_type == 'train':
        subset_size = math.ceil(n / duplication_level)  # Ensure subset_size is an integer
        data = load_data_from_file('train', 0, subset_size)
        if len(data) < subset_size:
            warnings.warn(f"Requested {subset_size} examples, but only {len(data)} examples available in train data.")

        labeled_data = [{'text': example['text'], 'label': 1} for example in data]
        labeled_data = labeled_data * duplication_level

        if len(labeled_data) > n:
            labeled_data = labeled_data[:n]  # Trim to exact requested number
        if duplication_level > 1:
            random.shuffle(labeled_data)

    elif data_type == 'test':
        data = load_data_from_file('test', 0, n)
        if len(data) < n:
            warnings.warn(f"Requested {n} examples, but only {len(data)} examples available in test data.")
        labeled_data = [{'text': example['text'], 'label': 1} for example in data]

    elif data_type == 'reserve':
        data = load_data_from_file('reserve', 0, n)
        if len(data) < n:
            warnings.warn(f"Requested {n} examples, but only {len(data)} examples available in reserve data.")
        labeled_data = [{'text': example['text'], 'label': 0} for example in data]

    elif data_type == 'train/reserve':
        subset_size = math.ceil(m / duplication_level)  # Ensure subset_size is an integer
        train_data = load_data_from_file('train', 0, subset_size)
        if len(train_data) < subset_size:
            warnings.warn(f"Requested {subset_size} examples, but only {len(train_data)} examples available in train data.")
        
        train_labeled = [{'text': example['text'], 'label': 1} for example in train_data]
        train_labeled = train_labeled * duplication_level
        if len(train_labeled) > m:
            train_labeled = train_labeled[:m]  # Trim to exact requested number

        reserve_data = load_data_from_file('reserve', reserve_offset, n)
        if len(reserve_data) < n:
            warnings.warn(f"Requested {n} examples with offset {reserve_offset}, but only {len(reserve_data)} examples available in reserve data after applying offset.")

        reserve_labeled = [{'text': example['text'], 'label': 0} for example in reserve_data]
        labeled_data = train_labeled + reserve_labeled
        random.shuffle(labeled_data)

    else:
        print("Invalid data type specified!")
        sys.exit()

    return labeled_data

def main():
    # Example usage
    num_train = 50
    num_test = 30
    num_reserve = 20
    reserve_offset = 10
    duplication_level = 2

    # Get train data
    train_data = get_data(data_type='train', n=num_train, duplication_level=duplication_level)
    print("Train Data:")
    for i in train_data[:10]:  # Print the first 10 examples
        print(i)

    # Get reserve data
    reserve_data = get_data(data_type='reserve', n=num_reserve)
    print("\nReserve Data:")
    for i in reserve_data[:10]:  # Print the first 10 examples
        print(i)

    # Get mixed train/reserve data with an offset for reserve data
    mixed_data = get_data(data_type='train/reserve', m=num_train, n=num_reserve, reserve_offset=reserve_offset, duplication_level=duplication_level)
    print("\nMixed Train/Reserve Data with Offset:")
    for i in mixed_data[:10]:  # Print the first 10 examples
        print(i)

if __name__ == "__main__":
    main()
