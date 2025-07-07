import cv2
from core.datasets import KittiMinimalDataset
import torch
import ptlflow
import numpy as np
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # NOTE: You must change this 'root_dir' to the actual path where your
    #       KITTI dataset is stored.
    try:
        root_dir = 'datasets/KITTI'
        print(f"Attempting to load dataset from: {root_dir}")

        # Instantiate the dataset
        kitti_dataset = KittiMinimalDataset(root=root_dir, split='training')

        if len(kitti_dataset) > 0:
            print(f"Successfully loaded {len(kitti_dataset)} training image pairs.")

            # --- Get a single sample from the dataset ---
            print("\nFetching the first sample (index 0) from the dataset...")
            sample = kitti_dataset[0]
            images_tensor = sample['images']

            print(f"Type of sample: {type(sample)}")
            print(f"Keys in sample: {sample.keys()}")
            print(f"Shape of 'images' tensor from dataset (2, C, H, W): {images_tensor.shape}")
            print(f"Data type of tensor: {images_tensor.dtype}")

            # --- Demonstrate usage with a DataLoader ---
            print("\n--- Simulating DataLoader ---")
            # A DataLoader with batch_size=4 would yield a dictionary where the
            # 'images' tensor has a shape of (4, 2, C, H, W).
            data_loader = DataLoader(dataset=kitti_dataset, batch_size=4, shuffle=False)

            # Get the first batch
            first_batch = next(iter(data_loader))
            batched_images_tensor = first_batch['images']

            print(f"Shape of 'images' tensor from DataLoader (B, 2, C, H, W): {batched_images_tensor.shape}")
            print("This is the exact format required by the ptlflow model.")

        else:
            print("Dataset loaded, but no samples were found.")

    except FileNotFoundError as e:
        print(e)
        print("\nPlease ensure the 'root_dir' variable points to the correct KITTI dataset location.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

