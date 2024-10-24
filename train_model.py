import torch
from torch import nn
from torchvision.models import vgg19
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from dataset import load_image_pairs  # Ensure this is correctly imported from your dataset.py
from utils import save_model, display_training_progress, EarlyStopping  # Import from utils.py
from PIL import Image
import os

# Set directory path in the local AI_3 directory for saving models
save_dir = '/content/AI_3/'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Exponential Moving Average (EMA) class
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """Register the initial parameters for the EMA model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Fixed syntax
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update the EMA model with new averaged weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Fixed syntax
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply the EMA weights to the model for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Fixed syntax
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights to the model after EMA evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Fixed syntax
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# UNet-like Generator model (ESRGAN)
class GeneratorUNetLike(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(GeneratorUNetLike, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Residual blocks (encoder)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # Upsampling layers with intermediate conv layers and skip connections
        self.upsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.refine1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # Conv layer after upsampling
        self.upsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # Conv layer after upsampling

        # Final output layer: Use tanh instead of sigmoid
        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        # Encoder
        x1 = self.leaky_relu(self.bn1(self.conv1(x)))
        x_res = self.residual_blocks(x1)

        # Upsampling + refinement (UNet-style)
        x_upsample1 = F.interpolate(x_res, scale_factor=2, mode='bilinear', align_corners=False)
        x_upsample1 = self.leaky_relu(self.upsample1(x_upsample1))
        x_upsample1 = self.leaky_relu(self.refine1(x_upsample1))  # Refine upsampled output

        x_upsample2 = F.interpolate(x_upsample1, scale_factor=2, mode='bilinear', align_corners=False)
        x_upsample2 = self.leaky_relu(self.upsample2(x_upsample2))
        x_upsample2 = self.leaky_relu(self.refine2(x_upsample2))  # Refine again after upsampling

        # Final output
        x_final = torch.tanh(self.conv2(x_upsample2))  # Use tanh instead of sigmoid
        return x_final

# Residual Block for Generator
class ResidualBlock(nn.Module):
    def __init__(self, filters=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return x + residual

# UNet-based Discriminator with Spectral Normalization (SN)
class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super().__init__()
        norm = spectral_norm
        self.skip_connection = skip_connection
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # Downsample layers
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # Upsample layers
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # Extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # Downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # Upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        # Extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

# VGG model for perceptual loss calculation
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(weights='IMAGENET1K_V1')  # Updated to use correct weights
        feature_layers = ['features.0', 'features.5', 'features.10', 'features.19', 'features.28']
        self.features = nn.ModuleList([vgg.features[int(layer.split('.')[1])] for layer in feature_layers])

    def forward(self, x):
        outputs = []
        for layer in self.features:
            x = layer(x)
            outputs.append(x)
        return outputs

# Perceptual loss function using VGG with weights {0.1, 0.1, 1, 1, 1}
def compute_perceptual_loss(vgg, y_true, y_pred):
    feature_weights = [0.1, 0.1, 1, 1, 1]
    loss = 0.0

    y_true_features = vgg(y_true)
    y_pred_features = vgg(y_pred)

    for i, weight in enumerate(feature_weights):
        loss += weight * nn.functional.mse_loss(y_pred_features[i], y_true_features[i])

    return loss

# Training step with generator and discriminator
def train_step(generator, discriminator, vgg, low_res_image, high_res_image, gen_optimizer, disc_optimizer, ema):
    generator.train()
    discriminator.train()

    # Move data to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    low_res_image = low_res_image.to(device)
    high_res_image = high_res_image.to(device)

    # Generator forward pass
    generated_image = generator(low_res_image)

    # Rescale generated image from [-1, 1] to [0, 1] for BCE loss
    generated_image_rescaled = (generated_image + 1) / 2

    # Discriminator forward pass
    real_output = discriminator(high_res_image)
    fake_output = discriminator(generated_image_rescaled.detach())

    # Compute perceptual loss
    perceptual_loss = compute_perceptual_loss(vgg, high_res_image, generated_image)

    # Adversarial loss: Using MSE for both real and fake losses
    adversarial_loss_real = nn.MSELoss()(real_output, torch.ones_like(real_output))  # Real images MSE loss
    adversarial_loss_fake = nn.MSELoss()(fake_output, torch.zeros_like(fake_output))  # Fake images MSE loss
    adversarial_loss = (adversarial_loss_real + adversarial_loss_fake) / 2

    # Total generator loss
    total_loss = perceptual_loss + adversarial_loss

    # Backpropagation for generator
    gen_optimizer.zero_grad()
    total_loss.backward()
    gen_optimizer.step()

    # EMA update
    ema.update()

    return total_loss.item(), generated_image

# Main training function with 850 epochs and updated datasets
def train_model():
    try:
        # Build the generator, discriminator, and VGG models
        generator = GeneratorUNetLike().cuda()
        discriminator = UNetDiscriminatorSN(num_in_ch=3).cuda()
        ema_generator = GeneratorUNetLike().cuda()  # EMA model for generator
        vgg = VGGFeatureExtractor().cuda()

        # Optimizers for generator and discriminator
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.9, 0.99))
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))

        # EMA for the generator
        ema = EMA(generator, decay=0.999)
        ema.register()

        # Load dataset from updated directories
        low_res_images, high_res_images = load_image_pairs('updated_low_res', 'updated_high_res', num_images=5)

        # Ensure directories exist locally
        os.makedirs('uploads', exist_ok=True)

        # Training loop for 850 epochs
        for epoch in range(850):
            for i, (low_res_image, high_res_image) in enumerate(zip(low_res_images, high_res_images)):
                low_res_image = torch.unsqueeze(low_res_image, 0)  # Add batch dimension
                high_res_image = torch.unsqueeze(high_res_image, 0)

                # Train the discriminator and generator
                combined_loss, generated_image = train_step(generator, discriminator, vgg, low_res_image, high_res_image, gen_optimizer, disc_optimizer, ema)

                print(f"Epoch {epoch + 1}, Image {i + 1}/{len(low_res_images)} - Loss: {combined_loss:.6f}")

                # Save the generated image locally to 'uploads'
                generated_image_np = generated_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                generated_image_np = (generated_image_np + 1) / 2  # Convert from [-1, 1] to [0, 1]
                generated_image_np = generated_image_np * 255.0    # Convert to [0, 255]
                generated_image_np = generated_image_np.clip(0, 255).astype('uint8')  # Clip and convert to uint8
                generated_image_pil = Image.fromarray(generated_image_np)
                generated_image_pil.save(f"uploads/generated_image_epoch_{epoch + 1}_image_{i + 1}.jpg")

                # Update EMA for the generator
                ema.update()

        # Apply EMA weights for the final generator model and save to the local directory
        ema.apply_shadow()
        torch.save(generator.state_dict(), f'{save_dir}/esrgan_final.pth')
        print(f"Final model saved locally at {save_dir}/esrgan_final.pth")

    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    train_model()
