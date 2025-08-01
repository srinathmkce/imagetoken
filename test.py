from datasets import load_dataset

# Load a dataset in streaming mode
dataset = load_dataset("some_dataset", split="train", streaming=True)

# Create batches of 32 samples
batched_dataset = dataset.batch(batch_size=32)

# Iterate over the batched dataset
for batch in batched_dataset:
    print(batch)
    break