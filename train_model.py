import argparse
import os
import torch
import torch.nn.functional as F  # Import torch.nn.functional for F.interpolate and F.leaky_relu
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import vgg19
from torch.nn.utils import spectral_norm
from torchvision.utils import save_image
from dataset import load_image_pairs, TrainDatasetFromFolder

# Create necessary directories
os.makedirs('images/training', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

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
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update the EMA model with new averaged weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply the EMA weights to the model for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights to the model after EMA evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# RRDB-Based Generator (Residual in Residual Dense Block)
class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]  # Corrected to nn.LeakyReLU
            return nn.Sequential(*layers)

        self.b1 = block(filters)
        self.b2 = block(2 * filters)
        self.b3 = block(3 * filters)
        self.b4 = block(4 * filters)
        self.b5 = block(5 * filters, non_linearity=False)

    def forward(self, x):
        inputs = x
        out1 = self.b1(inputs)
        out2 = self.b2(torch.cat([inputs, out1], 1))
        out3 = self.b3(torch.cat([inputs, out1, out2], 1))
        out4 = self.b4(torch.cat([inputs, out1, out2, out3], 1))
        out5 = self.b5(torch.cat([inputs, out1, out2, out3, out4], 1))
        return out5.mul(self.res_scale) + x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)

        # RRDB Blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])

        # Second convolutional layer
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2)
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

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

if __name__ == "__main__":

    # Argument Parsing for dynamic hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--batch_size', default=48, type=int, help='batch size of train dataset')
    parser.add_argument('--warmup_batches', default=1_000, type=int, help='number of batches with pixel-wise loss only')  # Reduced warmup to 1,000
    parser.add_argument('--n_batches', default=1_000, type=int, help='number of batches of training')
    parser.add_argument('--residual_blocks', default=23, type=int, help='number of residual blocks in the generator')
    parser.add_argument('--batch', default=0, type=int, help='batch to start training from')
    parser.add_argument('--lr', default=0.0002, type=float, help='adam: learning rate')
    parser.add_argument('--sample_interval', default=100, type=int, help='interval between saving image samples')
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize model parameters
    hr_shape = (opt.crop_size, opt.crop_size)
    channels = 3

    # Initialize the Generator and Discriminator
    generator = GeneratorRRDB(channels, num_res_blocks=opt.residual_blocks).to(device)
    discriminator = UNetDiscriminatorSN(channels).to(device)

    # Feature Extractor (VGG19) for perceptual loss
    feature_extractor = vgg19(weights='IMAGENET1K_V1').features[:29].to(device)
    feature_extractor.eval()  # Set VGG to eval mode

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)

    # Load models if resuming training
    if opt.batch != 0:
        generator.load_state_dict(torch.load('saved_models/generator_%d.pth' % opt.batch))
        discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth' % opt.batch))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # Initialize EMA for both generator and discriminator
    ema_G = EMA(generator, 0.999)
    ema_D = EMA(discriminator, 0.999)
    ema_G.register()
    ema_D.register()

    # Prepare dataset
    train_set = TrainDatasetFromFolder('updated_low_res', crop_size=opt.crop_size, upscale_factor=opt.upscale_factor)
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)  # Set num_workers=0

    # Training loop
    batch = opt.batch
    while batch < opt.n_batches:
        for i, (data, target) in enumerate(train_loader):
            batches_done = batch + i

            imgs_lr = data.to(device)
            imgs_hr = target.to(device)

            valid = torch.ones((imgs_lr.size(0), 1, *imgs_hr.shape[-2:]), requires_grad=False).to(device)
            fake = torch.zeros((imgs_lr.size(0), 1, *imgs_hr.shape[-2:]), requires_grad=False).to(device)

            # ---------------------
            # Training Generator
            # ---------------------

            optimizer_G.zero_grad()

            gen_hr = generator(imgs_lr)

            # Pixel-wise loss (L1 loss)
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            # Warmup phase: Only train generator with pixel loss
            if batches_done < opt.warmup_batches:
                loss_pixel.backward()
                optimizer_G.step()
                ema_G.update()
                print(f"[Warmup] [Iteration {batches_done}/{opt.n_batches}] [Batch {i}/{len(train_loader)}] [G pixel: {loss_pixel.item():.6f}]")
                continue
            elif batches_done == opt.warmup_batches:
                optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)

            # GAN Loss
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Relativistic GAN Loss
            loss_GAN = (
                               criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid) +
                               criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)
                       ) / 2

            # Perceptual Loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            real_features = [real_f.detach() for real_f in real_features]
            loss_content = sum(criterion_content(gen_f, real_f) * w for gen_f, real_f, w in zip(gen_features, real_features, [0.1, 0.1, 1, 1, 1]))

            # Total Generator Loss: Pixel loss + Perceptual loss + GAN loss
            loss_G = loss_content + 0.1 * loss_GAN + loss_pixel

            loss_G.backward()
            optimizer_G.step()
            ema_G.update()

            # ---------------------
            # Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Relativistic GAN Loss for Discriminator
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()
            ema_D.update()

            # -------------------------
            # Log Progress
            # -------------------------

            print(f"[Iteration {batches_done}/{opt.n_batches}] [Batch {i}/{len(train_loader)}] [D loss: {loss_D.item():.6f}] [G loss: {loss_G.item():.6f}, content: {loss_content.item():.6f}, adv: {loss_GAN.item():.6f}, pixel: {loss_pixel.item():.6f}]")

            # Save image samples every opt.sample_interval iterations
            if batches_done % opt.sample_interval == 0:
                imgs_lr = F.interpolate(imgs_lr, scale_factor=4, mode='bicubic')
                img_grid = torch.clamp(torch.cat((imgs_lr, gen_hr, imgs_hr), -1), min=0, max=1)
                save_image(img_grid, 'images/training/%d.png' % batches_done, nrow=1, normalize=False)

        batch = batches_done + 1

        # Save model and EMA weights
        ema_G.apply_shadow()
        ema_D.apply_shadow()
        torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % batch)
        torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % batch)
        ema_G.restore()
        ema_D.restore()
