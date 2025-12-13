# Investigating the Impact of Distributionally Robust Optimization on Learning with Noisy Labels

**COMP-5422-FA - Comp Vision & Image Analysis Course Project** 

This repository contains PyTorch implementations for two distinct machine learning research tracks:
1. **Distributionally Robust Optimization (DRO):** Handling spurious correlations and group shifts, specifically tested on the Waterbirds dataset.
2. **Learning with Noisy Labels:** Training robust models on the CIFAR dataset with symmetric and asymmetric label noise (implementing a DivideMix-style pipeline).

## Project Structure

### 1. Waterbirds / DRO (Distributionally Robust Optimization)
Files related to training robust models that minimize worst-group error.
* `Train_dro_run_exep.py`: Main entry point for training the DRO model (ResNet50) on the Waterbirds dataset.
* `dro_dataset.py`: Defines the `DRODataset` wrapper, handling group metadata and counting for robust optimization.
* `dro_loss.py`: Implements the `LossComputer` class, which calculates standard ERM loss, robust loss, and tracks group-wise statistics.
* `dro_train.py`: Contains the training loop (`run_epoch`), handling forward passes, gradient updates, and logging.
* `dro_data.py`: Utilities for splitting the dataset into train/val/test and handling confounder splits.
* `dro_utils.py`: Helper functions for logging (CSV/Console) and seeding.
* `dataset_waterbird_visualization.ipynb`: Jupyter notebook for analyzing the Waterbirds dataset (e.g., t-SNE visualizations of embeddings).

### 2. CIFAR / Noisy Labels
Files related to training ResNet models in the presence of label noise.
* `Train_cifar.py`: Main training script for CIFAR-10/100. Implements semi-supervised learning techniques (co-divide, noise thresholding) to handle noisy labels.
* `dataloader_cifar.py`: Custom dataloader that injects symmetric or asymmetric noise into the labels.
* `dataset_noisy_cifar_visualization.ipynb`: Jupyter notebook for visualizing the confusion matrices between clean and noisy labels and displaying sample images.

## 🛠 Installation & Requirements

Ensure you have Python 3.x installed along with the following dependencies:
    
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm torchnet
```
_Note: The code assumes a CUDA-capable GPU is available._

## Usage
### Part 1: DRO on Waterbirds
The DRO module expects the Waterbirds dataset to be located in `../datasets/waterbird` (configurable in `Train_dro_run_exep.py`).

To run the training script:

```bash
python Train_dro_run_exep.py
```
#### Key Parameters (inside `main`):

- `robust`: Set to `True` to enable Distributionally Robust Optimization.

- `alpha`: Hyperparameter for the robust loss.

- `reweight_groups`: Enable importance weighting for groups.

### Part 2: Noisy CIFAR Training
The CIFAR training script allows you to specify the noise mode and intensity.

To train on CIFAR-10 with 50% symmetric noise:

```bash
python Train_cifar.py --batch_size 128 --lr 0.02 --noise_mode sym --r 0.5 --num_epochs 10
```
#### Key Arguments:

- `--r`: Noise ratio (e.g., `0.5` for 50% noise).

- `--noise_mode`: Type of noise (`sym` for symmetric, `asym` for asymmetric).

- `--p_threshold`: Clean probability threshold for dividing labeled/unlabeled data.

- `--lambda_u`: Weight for unsupervised loss.

## Visualizations
This repository includes Jupyter notebooks to help visualize the data distributions:

1. `dataset_noisy_cifar_visualization.ipynb`:

- Visualizes the transition matrix of noisy labels.

- Displays sample images alongside their clean and noisy labels.

2. `dataset_waterbird_visualization.ipynb`:

- Visualizes the Waterbirds dataset using t-SNE on ResNet embeddings.

- Highlights the separation between background and foreground (bird) classes.