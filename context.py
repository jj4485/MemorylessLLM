from datasets import load_dataset

# Stream the dataset
dataset = load_dataset("the_pile", split="train", streaming=True)

# Take a sample, for example, the first 10000 examples
sample = []
for i, example in enumerate(dataset):
    sample.append(example)
    if i >= 10000:
        break