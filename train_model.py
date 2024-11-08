import torch
from torch import nn
from torchvision.models import vgg19
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from dataset import load_image_pairs  # Ensure this is correctly imported from your dataset.py
from utils import save_model, display_training_progress, EarlyStopping  # Import from utils.py
from PIL import Image
import os

# Set directory path in a writable location on your local machine
save_dir = '/Users/sachinrao/complex_proj/AI_3/'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Exponential Moving Average (EMA) class
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# Generator with RRDB architecture for ESRGAN
class GeneratorRRDB(nn.Module):
    def __init__(self, num_residual_blocks=23):
        super(GeneratorRRDB, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.residual_blocks = nn.Sequential(
            *[RRDBBlock(64) for _ in range(num_residual_blocks)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_blocks(x) + x
        x = self.upsample(x)
        return torch.tanh(x)

# RRDB and RDB blocks
class RRDBBlock(nn.Module):
    def __init__(self, filters):
        super(RRDBBlock, self).__init__()
        self.block1 = ResidualDenseBlock(filters)
        self.block2 = ResidualDenseBlock(filters)
        self.block3 = ResidualDenseBlock(filters)

    def forward(self, x):
        return self.block1(x) + self.block2(x) + self.block3(x)

class ResidualDenseBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layers(x) + x

# Load ESRGAN pretrained weights
def load_pretrained_generator(generator, weights_path):
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        generator.load_state_dict(checkpoint, strict=False)
    return generator

# Training function
def train_model():
    try:
        generator = GeneratorRRDB().cuda()
        discriminator = UNetDiscriminatorSN(num_in_ch=3).cuda()
        ema_generator = GeneratorRRDB().cuda()
        vgg = VGGFeatureExtractor().cuda()

        # Load pretrained ESRGAN weights
        generator = load_pretrained_generator(generator, '/Users/sachinrao/complex_proj/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth')

        # Optimizers
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.9, 0.99))
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))

        # EMA
        ema = EMA(generator, decay=0.999)
        ema.register()

        # Load dataset
        low_res_images, high_res_images = load_image_pairs('low_res', 'high_res', num_images=854)

        # Training loop
        for epoch in range(209):
            for i, (low_res_image, high_res_image) in enumerate(zip(low_res_images, high_res_images)):
                low_res_image = torch.unsqueeze(low_res_image, 0).cuda()
                high_res_image = torch.unsqueeze(high_res_image, 0).cuda()

                # Train step
                combined_loss, generated_image = train_step(generator, discriminator, vgg, low_res_image, high_res_image, gen_optimizer, disc_optimizer, ema)
                print(f"Epoch {epoch + 1}, Image {i + 1}/{len(low_res_images)} - Loss: {combined_loss:.6f}")

                # Save generated images
                generated_image_np = generated_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255
                generated_image_np = generated_image_np.clip(0, 255)
                generated_image_pil = Image.fromarray(generated_image_np.astype('uint8'))
                generated_image_pil.save(f"{save_dir}/uploads/generated_image_epoch_{epoch + 1}_image_{i + 1}.jpg")

                # EMA update
                ema.update()

        # Final model save
        ema.apply_shadow()
        torch.save(generator.state_dict(), f'{save_dir}/esrgan_final.pth')
        print(f"Final model saved locally at {save_dir}/esrgan_final.pth")

    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    train_model()
