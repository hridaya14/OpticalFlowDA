import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class KittiCleanFoggyDataset(Dataset):
    """
    A PyTorch Dataset that loads sequential pairs of clean images and their
    corresponding pre-generated foggy versions from the KITTI dataset.

    This loader returns a dictionary containing two tensors:
    - 'clean_images': A tensor of the clean image pair with shape (2, C, H, W).
    - 'foggy_images': A tensor of the foggy image pair with shape (2, C, H, W).

    Args:
        root (str): The root directory of the clean KITTI dataset.
        foggy_root (str): The directory where pre-generated foggy images are stored.
        split (str): The dataset split, typically 'training'.
    """
    def __init__(self, root='datasets/KITTI', foggy_root='generate_images', split='training'):
        super(KittiCleanFoggyDataset, self).__init__()

        # --- 1. Set up file paths ---
        self.split = split
        clean_image_root = os.path.join(root, split, 'image_2')

        if not os.path.isdir(clean_image_root):
            raise FileNotFoundError(f"Clean image directory not found at: {clean_image_root}")
        if not os.path.isdir(foggy_root):
            raise FileNotFoundError(f"Foggy image directory not found at: {foggy_root}")

        # --- 2. Find all corresponding image sets ---
        self.samples = []
        images1 = sorted(glob.glob(os.path.join(clean_image_root, '*_10.png')))

        for img1_path in images1:
            # Construct paths for the clean pair
            img2_path = img1_path.replace('_10.png', '_11.png')

            # Construct paths for the foggy pair
            basename = os.path.basename(img1_path)
            foggy1_path = os.path.join(foggy_root, basename)
            foggy2_path = foggy1_path.replace('_10.png', '_11.png')

            # Ensure all four images exist before adding to the list
            if all(os.path.exists(p) for p in [img1_path, img2_path, foggy1_path, foggy2_path]):
                self.samples.append({
                    "clean1": img1_path, "clean2": img2_path,
                    "foggy1": foggy1_path, "foggy2": foggy2_path
                })

        if not self.samples:
            print("Warning: No complete sets of clean and foggy images were found.")
        else:
            print(f"Found {len(self.samples)} complete sets of clean and foggy image pairs.")


    def __getitem__(self, index):
        """
        Retrieves a set of clean and foggy image pairs from the dataset.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing 'clean_images' and 'foggy_images' tensors.
        """
        paths = self.samples[index]

        # --- Keep original logic for clean images ---
        clean1_np = np.array(Image.open(paths['clean1']).convert('RGB')).astype(np.uint8)
        clean2_np = np.array(Image.open(paths['clean2']).convert('RGB')).astype(np.uint8)

        clean1_tensor = torch.from_numpy(clean1_np).permute(2, 0, 1).float()
        clean2_tensor = torch.from_numpy(clean2_np).permute(2, 0, 1).float()

        clean_images = torch.stack([clean1_tensor, clean2_tensor], dim=0)

        # --- Apply identical logic for foggy images ---
        foggy1_np = np.array(Image.open(paths['foggy1']).convert('RGB')).astype(np.uint8)
        foggy2_np = np.array(Image.open(paths['foggy2']).convert('RGB')).astype(np.uint8)

        foggy1_tensor = torch.from_numpy(foggy1_np).permute(2, 0, 1).float()
        foggy2_tensor = torch.from_numpy(foggy2_np).permute(2, 0, 1).float()

        foggy_images = torch.stack([foggy1_tensor, foggy2_tensor], dim=0)

        return {'clean_images': clean_images, 'foggy_images': foggy_images}

    def __len__(self):
        """
        Returns the total number of image sets in the dataset.
        """
        return len(self.samples)
