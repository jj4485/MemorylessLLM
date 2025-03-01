from datasets import load_dataset

dataset = load_dataset("the_pile", split="train", streaming=True)

# Iterate over examples one by one
for example in dataset:
    # Process each example as it comes
    print(example)
    break