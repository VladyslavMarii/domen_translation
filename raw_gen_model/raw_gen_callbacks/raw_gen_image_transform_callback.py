import tensorflow as tf
import os
import random
from tf_keras.callbacks import Callback
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt

class ImageTransformCallback(Callback):
    def __init__(self, validation_data, encoder, decoder, save_dir='./reconstructed_images'):
        super().__init__()
        # Store the validation data (list/tuple of pairs, or just images if you prefer).
        self.validation_data = list(validation_data)
        self.encoder = encoder
        self.decoder = decoder

        # Directory to save the .png files
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def set_model(self, model):
        super().set_model(model)
        
    def on_epoch_end(self, epoch, logs=None):
        # Pick a random sample from the validation dataset
        random_index = random.randint(0, len(self.validation_data) - 1)
        satellite_image, drone_image = self.validation_data[random_index]

        # Ensure both images are batched
        if len(satellite_image.shape) == 3:
            satellite_image = tf.expand_dims(satellite_image, axis=0)
        if len(drone_image.shape) == 3:
            drone_image = tf.expand_dims(drone_image, axis=0)

        # Encode & decode
        tokens = self.encoder(drone_image)
        transformed_tokens, _ = self.model(tokens)
        reconstructed_image = self.decoder(transformed_tokens)

        # Convert to PIL images
        satellite_pil = tf.keras.preprocessing.image.array_to_img(satellite_image[0])
        drone_pil = tf.keras.preprocessing.image.array_to_img(drone_image[0])
        reconstructed_pil = tf.keras.preprocessing.image.array_to_img(reconstructed_image[0])

        # Display side by side
        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        ax[0].imshow(drone_pil)
        ax[0].set_title("Drone")
        ax[1].imshow(reconstructed_pil)
        ax[1].set_title("Reconstructed")
        ax[2].imshow(satellite_pil)
        ax[2].set_title("Satellite")

        for a in ax:
            a.axis("off")

        # Save reconstructed image
        output_path = os.path.join(self.save_dir, f'reconstructed_epoch_{epoch}_idx_{random_index}.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.show()

        print(f"[SatelliteReconstructionCallback] Saved reconstructed image to: {output_path}")