import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, BatchNormalization, Add, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
import numpy as np
from PIL import Image
import os

# Define the residual block for ESRGAN with Dropout for regularization
def residual_block(input_layer, filters=64):
    x = Conv2D(filters, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)  # Dropout to prevent overfitting
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    return Add()([input_layer, x])

# Build the ESRGAN-like generator model with residual blocks
def build_generator():
    input_img = Input(shape=(128, 128, 3))  # Low-res image input

    # First convolution block
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    x = BatchNormalization()(x)

    # Add residual blocks
    for _ in range(16):  # 16 residual blocks for more feature learning
        x = residual_block(x)

    # First upsampling block
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Second upsampling block
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Final convolutional layer to output high-resolution image
    output_img = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

    return Model(inputs=input_img, outputs=output_img)

# Build the discriminator to evaluate whether the generated images are realistic
def build_discriminator():
    input_img = Input(shape=(512, 512, 3))  # High-res image input

    # Convolutional block
    x = Conv2D(64, (3, 3), padding='same', strides=2)(input_img)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    # Add more convolutional layers
    for _ in range(4):
        x = Conv2D(64, (3, 3), padding='same', strides=2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

    # Final output layer for binary classification (real or fake)
    output = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)

    return Model(inputs=input_img, outputs=output)

# Build the VGG model for perceptual loss calculation
def build_vgg_for_loss():
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    model = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
    model.trainable = False  # Freeze the VGG model for feature extraction
    return model

# Perceptual loss function using VGG
@tf.function
def compute_perceptual_loss(vgg, y_true, y_pred):
    true_features = vgg(y_true)  # Ensure y_true is tensor
    pred_features = vgg(y_pred)  # Ensure y_pred is tensor
    return tf.reduce_mean(tf.square(true_features - pred_features))

# Training step wrapped in @tf.function to optimize memory usage
@tf.function
def train_step(generator, discriminator, vgg, low_res_image, high_res_image, real_labels, fake_labels, gen_optimizer, disc_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        # Generate high-resolution images using the generator
        generated_image = generator(low_res_image)

        # Discriminator predictions
        real_preds = discriminator(high_res_image)
        fake_preds = discriminator(generated_image)

        # Calculate losses
        mse_loss = tf.reduce_mean(tf.square(high_res_image - generated_image))  # Pixel-wise MSE loss
        perceptual_loss_value = compute_perceptual_loss(vgg, high_res_image, generated_image)  # Perceptual loss
        adversarial_loss_value = tf.reduce_mean(fake_preds - real_preds)  # Adversarial loss

        combined_loss = mse_loss + perceptual_loss_value + adversarial_loss_value

        # Discriminator loss (binary cross-entropy)
        real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_preds)
        fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_preds)
        disc_loss = real_loss + fake_loss

    # Apply gradients to update generator and discriminator
    gen_gradients = tape.gradient(combined_loss, generator.trainable_variables)
    disc_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return combined_loss, generated_image

# Function to load and resize images
def load_image(image_path, target_size=None):
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return None
    img = Image.open(image_path).convert('RGB')  # Convert to RGB mode to ensure 3 channels
    if target_size:
        img = img.resize(target_size, Image.BICUBIC)  # Resize using bicubic interpolation
    img = np.array(img) / 255.0  # Normalize image to [0, 1]
    return tf.convert_to_tensor(img, dtype=tf.float32)  # Convert image to tensor

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

    return tf.stack(low_res_images), tf.stack(high_res_images)

# Main training function with early stopping and checkpoints
def train_model():
    try:
        # Build the generator, discriminator, and VGG models
        generator = build_generator()
        discriminator = build_discriminator()
        vgg = build_vgg_for_loss()

        # Optimizers for generator and discriminator
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

        # Directories for low-res and high-res images
        low_res_dir = 'low_res'  # Directory containing low-res images
        high_res_dir = 'high_res'  # Directory containing high-res images

        # Load all low-res and high-res image pairs (685 images)
        print("Loading image pairs for training...")
        low_res_images, high_res_images = load_image_pairs(low_res_dir, high_res_dir, num_images=685)

        # Get the output shape of the discriminator to create correct label sizes
        disc_output_shape = discriminator.output_shape[1:4]  # Get the spatial dimensions of the discriminator's output

        # Labels for real and fake images
        real_labels = tf.ones((1, *disc_output_shape))  # Create labels that match discriminator output
        fake_labels = tf.zeros((1, *disc_output_shape))

        previous_loss = float('inf')  # Initialize previous loss for early stopping
        early_stopping_threshold = 1e-4  # Threshold to trigger early stopping

        # Train the model on this set of image pairs (for 500 epochs)
        for epoch in range(500):  # Train for 500 epochs
            # Create a directory for each epoch to store generated images
            os.makedirs(f"uploads/epoch_{epoch+1}", exist_ok=True)

            for i in range(len(low_res_images)):
                low_res_image = tf.expand_dims(low_res_images[i], axis=0)  # Expand dimension for a single image
                high_res_image = tf.expand_dims(high_res_images[i], axis=0)

                # Perform a training step
                combined_loss, generated_image = train_step(generator, discriminator, vgg, low_res_image, high_res_image, real_labels, fake_labels, gen_optimizer, disc_optimizer)

                print(f"Epoch {epoch + 1}/500, Image {i + 1}/685 - Combined Loss: {combined_loss:.6f}")

                # Convert and save the generated image (outside @tf.function)
                generated_image_np = generated_image.numpy()  # Convert to NumPy array
                generated_image_np = tf.squeeze(generated_image_np) * 255.0  # Scale back to [0, 255]
                generated_image_np = tf.clip_by_value(generated_image_np, 0, 255).numpy().astype(np.uint8)
                result_img = Image.fromarray(generated_image_np)
                result_img.save(f"uploads/epoch_{epoch+1}/generated_image_{i+1}.jpg")

            # Save checkpoints for both generator and discriminator
            generator.save(f'checkpoints/epoch_{epoch+1}_generator.h5')
            discriminator.save(f'checkpoints/epoch_{epoch+1}_discriminator.h5')

            # Early stopping based on the combined loss
            if epoch > 10 and abs(previous_loss - combined_loss) < early_stopping_threshold:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            previous_loss = combined_loss  # Update the previous loss

        # Final save for the generator model
        generator.save('esrgan_generator.h5')
        print("Model training complete and saved as 'esrgan_generator.h5'")

    except Exception as e:
        print(f"Error during image processing or training: {str(e)}")

# Call the function to train and save the model
if __name__ == "__main__":
    train_model()
