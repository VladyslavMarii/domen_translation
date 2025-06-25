import json
from pathlib import Path
import tensorflow as tf
from tf_keras.callbacks import Callback
import random

from config import LOGS_DIR

class TokenMatchingCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = list(validation_data)
        
    def set_model(self, model):
        super().set_model(model)

    def set_encoder(self, encoder):
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        random_index = random.randint(0, len(self.validation_data) - 1)
        satellite_image, drone_image = self.validation_data[random_index]

        # Obtain drone tokens, translated tokens, and satellite tokens
        drone_tokens = self.encoder(drone_image)
        translated_tokens, _ = self.model(drone_tokens, training=True)
        satelite_tokens = self.encoder(satellite_image)

        rand_idx_for_comparing = random.randint(0, drone_tokens.shape[1] - 1)

        # Manhattan distance for the same index
        distance_same_index = tf.reduce_sum(
            tf.abs(
                satelite_tokens[0, rand_idx_for_comparing] - translated_tokens[0, rand_idx_for_comparing]
            )
        )
        print(f"Manhattan distance at the same index ({rand_idx_for_comparing}): "
            f"{distance_same_index.numpy()}")

        # Manhattan distance for all tokens in translated_tokens[0]
        all_distances = tf.reduce_sum(
            tf.abs(
                satelite_tokens[0, rand_idx_for_comparing] - translated_tokens[0, :]
            ), 
            axis=1  # sum over embedding dimension
        )

        # Find the best (lowest) distance and its index
        best_idx = tf.argmin(all_distances)
        best_distance = all_distances[best_idx]

        print(f"Best (lowest) Manhattan distance index: {best_idx.numpy()}, "
            f"distance: {best_distance.numpy()}")
        val_loss = logs.get('val_loss')
        data = {
            "epoch": epoch + 1,
            "random_index": random_index,
            "rand_idx_for_comparing": int(rand_idx_for_comparing),
            "distance_same_index": float(distance_same_index.numpy()),
            "best_idx": int(best_idx.numpy()),
            "best_distance": float(best_distance.numpy()),
        }
        if val_loss is not None:
            data["val_loss"] = float(val_loss)
        folder_path = Path(f"{LOGS_DIR}/manchatan")
        folder_path.mkdir(parents=True, exist_ok=True)
        save_path = folder_path / f"mantoken_distance_log_epoch_{epoch + 1}.json"        
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved distance info to {save_path}")

