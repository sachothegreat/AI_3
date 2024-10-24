import os
import torch
import torch.nn.functional as F  # Import torch.nn.functional for interpolation
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# Function to load and resize images
def load_image(image_path, target_size=None):
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return None
    img = Image.open(image_path).convert('RGB')  # Convert to RGB mode
    if target_size:
        img = img.resize(target_size, Image.BICUBIC)  # Resize using bicubic interpolation
    img = np.array(img) / 255.0  # Normalize image to [0, 1]
    img = img * 2 - 1  # Rescale image from [0, 1] to [-1, 1]
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and adjust dimensions

# Function to load all image pairs (low and high res) for training
def load_image_pairs(low_res_dir, high_res_dir, num_images):
    low_res_images = []
    high_res_images = []

    for i in range(num_images):
        low_res_path = os.path.join(low_res_dir, f'{i}.png')
        high_res_path = os.path.join(high_res_dir, f'{i}.png')

        low_res_image = load_image(low_res_path, target_size=(128, 128))
        high_res_image = load_image(high_res_path, target_size=(512, 512))

        if low_res_image is not None and high_res_image is not None:
            low_res_images.append(low_res_image)
            high_res_images.append(high_res_image)
        else:
            print(f"Skipping pair: {i}.png")

    return torch.stack(low_res_images), torch.stack(high_res_images)

# Custom dataset class to load training images from a folder
class TrainDatasetFromFolder(Dataset):
    def __init__(self, folder, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(folder, x) for x in os.listdir(folder) if self.is_image_file(x)]
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index]).convert('RGB')

        # High-resolution image
        hr_image = self.transform_hr(img)

        # Low-resolution image
        lr_image = self.transform_lr(hr_image)

        return lr_image, hr_image

    def transform_hr(self, img):
        """Transform high-resolution image."""
        # Crop to target size
        hr_image = img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        hr_image = np.array(hr_image) / 255.0
        hr_image = hr_image * 2 - 1  # Rescale to [-1, 1]
        return torch.tensor(hr_image, dtype=torch.float32).permute(2, 0, 1)

    def transform_lr(self, hr_image):
        """Transform low-resolution image."""
        # Use torch.nn.functional.interpolate to downsample the high-resolution image
        lr_image = F.interpolate(hr_image.unsqueeze(0), scale_factor=1/self.upscale_factor, mode='bicubic', align_corners=False)
        return lr_image.squeeze(0)

    def __len__(self):
        return len(self.image_filenames)
