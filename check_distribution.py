import numpy as np
import dataloader_cifar as dataloader
import sys

# Configuration (Same as your training script)
DATASET = 'cifar10'
ROOT_DIR = './cifar-10-batches-py'
NOISE_RATIO = 0.1   # Change this to check different noise levels
NOISE_MODE = 'sym'
BATCH_SIZE = 64

# Initialize the Dataloader Wrapper
print(f"Loading Data (Noise: {NOISE_RATIO}, Mode: {NOISE_MODE})...")
loader_wrapper = dataloader.cifar_dataloader(
    dataset=DATASET,
    r=NOISE_RATIO,
    noise_mode=NOISE_MODE,
    batch_size=BATCH_SIZE,
    num_workers=0,
    root_dir=ROOT_DIR,
    log=sys.stdout,  # Print logs to console
    noise_file=f'./{NOISE_RATIO}_{NOISE_MODE}_noise.json' # dummy path
)

# 1. Get Training Distribution (using 'warmup' mode which loads ALL data)
print("\n--- Training Set Distribution ---")
warmup_loader = loader_wrapper.run('warmup')
train_dataset = warmup_loader.dataset

# Access internal data stored in the dataset class
# Note: In your dataloader_cifar_2.py:
# - self.binary_label stores the NOISY binary labels (used for training)
# - self.original_label stores the CLEAN multiclass labels (0-9)

# A. Check Noisy Binary Labels (What the model sees)
noisy_binary = np.array(train_dataset.binary_label)
noisy_birds = (noisy_binary == 1).sum()
noisy_rest = (noisy_binary == 0).sum()
total_train = len(noisy_binary)

print(f"Total Training Images: {total_train}")
print(f"Noisy 'Bird' Labels (Class 1): {noisy_birds} ({noisy_birds/total_train:.2%})")
print(f"Noisy 'Rest' Labels (Class 0): {noisy_rest} ({noisy_rest/total_train:.2%})")

# B. Check True Clean Labels (Ground Truth)
# We need to manually convert multiclass original_label (0-9) to binary (Bird=2 -> 1)
clean_multiclass = np.array(train_dataset.original_label)
clean_binary = (clean_multiclass == 2).astype(int) # Class 2 is Bird in CIFAR-10
clean_birds = (clean_binary == 1).sum()
clean_rest = (clean_binary == 0).sum()

print(f"\nTrue 'Bird' Images (Ground Truth): {clean_birds} ({clean_birds/total_train:.2%})")
print(f"True 'Rest' Images (Ground Truth): {clean_rest} ({clean_rest/total_train:.2%})")

# 2. Get Test Set Distribution
print("\n--- Test Set Distribution ---")
test_loader = loader_wrapper.run('test')
test_dataset = test_loader.dataset

# In test mode, self.test_label contains the binary labels
test_binary = np.array(test_dataset.test_label)
test_birds = (test_binary == 1).sum()
test_rest = (test_binary == 0).sum()
total_test = len(test_binary)

print(f"Total Test Images: {total_test}")
print(f"Test 'Bird' Labels: {test_birds} ({test_birds/total_test:.2%})")
print(f"Test 'Rest' Labels: {test_rest} ({test_rest/total_test:.2%})")