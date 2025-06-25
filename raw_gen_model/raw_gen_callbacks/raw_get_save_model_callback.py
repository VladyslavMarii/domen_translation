import os
from pathlib import Path

import numpy as np
import tf_keras

from config import LOGS_DIR, TRAINING_ID


class SaveBestRawModelCallback(tf_keras.callbacks.Callback):
    """
    Saves drone_encoder and satellite_encoder whenever 'monitor' metric improves.
    """

    def __init__(self, model, monitor: str = 'val_loss', mode: str = 'min'):
        """
        Args:
            drone_encoder: The DroneEncoder model instance.
            satellite_encoder: The SatelliteEncoder model instance.
            monitor: Metric to monitor (e.g. 'val_loss', 'val_accuracy').
            mode: One of 'min' or 'max'. In 'min' mode,
                  we save when the metric decreases;
                  in 'max' mode, when it increases.
            save_path: Directory path in which to save the models.
        """
        super().__init__()
        self.model = model  
        self.monitor = monitor
        self.mode = mode

        if mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'.")

        self.best = np.inf if mode == 'min' else -np.inf
        self.saved_models = [] 

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = True # (
        #     (self.mode == 'min' and current < self.best) or
        #     (self.mode == 'max' and current > self.best)
        # )
        if improved:
            self.best = current
            folder_path = Path(f"{LOGS_DIR}/checkpoints")
            folder_path.mkdir(parents=True, exist_ok=True)
            model_path = f"{folder_path}/epoch{epoch+1}_{self.monitor}_{current:.4f}.keras"
            
            # Save current model
            if self.model is not None:
                self.model.save(model_path)
                print(
                    f"Epoch {epoch+1}: new best {self.monitor} = {current:.4f}. "
                    f"Saved model to {model_path}"
                )
                # Track saved model
                # self.saved_models.append((model_path, current))
                self.saved_models.append((model_path, epoch+1))
                # If more than 3 saved, remove the worst
                if len(self.saved_models) > 3:
                    if self.mode == 'min':
                        # worst = max metric
                        worst = max(self.saved_models, key=lambda x: x[1])
                    else:
                        # worst = min metric
                        worst = min(self.saved_models, key=lambda x: x[1])

                    # -----------------------------------------
                    worst = min(self.saved_models, key=lambda x: x[1])
                    # -----------------------------------------

                    self.saved_models.remove(worst)
                    if os.path.exists(worst[0]):
                        os.remove(worst[0])
                        print(f"Removed worst model: {worst[0]}")
            else:
                print("Warning: self.model is None. Cannot save.")
