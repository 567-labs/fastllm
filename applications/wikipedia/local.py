from datasets import load_dataset, load_from_disk
import os

data_dir = "data"
cache_dir = os.path.join(os.getcwd(), data_dir)
print(cache_dir)
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Step 1: Download the 'squad' dataset and cache it in a specified directory
dataset = load_dataset("squad", cache_dir=cache_dir.__str__())
import pdb

pdb.set_trace()
# # Print a message to indicate that the dataset has been downloaded and cached
# print("Dataset downloaded and cached.")

# # Step 2: Save the dataset to disk
# # The dataset is already saved to disk in the previous step when specifying the 'cache_dir' parameter
# # Step 2: Save the dataset to disk
# dataset.save_to_disk(cache_dir.__str__())

# # Step 3: Load the dataset from the specified directory
# loaded_dataset = load_from_disk(cache_dir.__str__())

# # Step 3: Load the dataset from the specified directory
# loaded_dataset = load_from_disk(cache_dir.__str__())

# # Print a message to indicate that the dataset has been loaded
# print("Dataset loaded from disk.")

# # Print the loaded dataset
# print(loaded_dataset)
