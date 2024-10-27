import os
import torch
from PIL import Image
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
