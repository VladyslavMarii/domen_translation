import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2  # OpenCV for resizing
import random
from tf_keras.callbacks import Callback

from helper import feature_norm

class AttentionVisualizationCallback(Callback):
    def __init__(self, validation_data, log_dir="attention_maps", save_every_n_epochs=1):
        super().__init__()
        self.validation_data = list(validation_data)
        self.log_dir = log_dir  # Use consistent naming
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(log_dir, exist_ok=True)
        
        self.patch_size = 16  # Should match encoder's patch_size
        self.H = self.W = 400 // self.patch_size 
    def _save_attention_visualization(self, anchor, positive, attn_pos, epoch):
        # Debug prints to check shapes
        print(f"Anchor shape: {anchor.shape}")
        print(f"Positive shape: {positive.shape}")
        print(f"Attention map shape: {attn_pos.shape}")

        # Take first item from batch
        anchor = anchor[0]  # Remove batch dimension
        positive = positive[0]  # Remove batch dimension
        attn_pos = attn_pos[0]  # Remove batch dimension

        # Normalize attention map to 0-1 range if needed
        attn_pos = (attn_pos - attn_pos.min()) / (attn_pos.max() - attn_pos.min() + 1e-8)
        
        # Reshape attention map to match image dimensions
        attn_pos = cv2.resize(attn_pos, (anchor.shape[1], anchor.shape[0]))
        
        # Convert attention map to heatmap
        heatmap_pos = cv2.applyColorMap(np.uint8(255 * attn_pos), cv2.COLORMAP_JET)
        
        # Convert input images to uint8 and ensure 3 channels
        def prepare_image(img):
            # Convert to uint8 if not already
            img_uint8 = np.uint8(img * 255) if img.dtype != np.uint8 else img
            # Ensure 3 channels
            if len(img_uint8.shape) == 2:
                img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
            elif len(img_uint8.shape) == 3 and img_uint8.shape[2] == 1:
                img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
            return img_uint8

        anchor_uint8 = prepare_image(anchor)
        positive_uint8 = prepare_image(positive)

        # Create overlays
        overlay_anchor_pos = cv2.addWeighted(anchor_uint8, 0.6, heatmap_pos, 0.4, 0)
        overlay_positive = cv2.addWeighted(positive_uint8, 0.6, heatmap_pos, 0.4, 0)

        # Convert BGR to RGB for matplotlib
        overlay_anchor_pos = cv2.cvtColor(overlay_anchor_pos, cv2.COLOR_BGR2RGB)
        overlay_positive = cv2.cvtColor(overlay_positive, cv2.COLOR_BGR2RGB)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        
        axes[0].imshow(overlay_anchor_pos)
        axes[0].set_title("Anchor + Positive Attention")
        axes[1].imshow(overlay_positive)
        axes[1].set_title("Positive + Positive Attention")
        
        for ax in axes.flat:
            ax.axis("off")
        
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="jet"), cax=cbar_ax)
        cbar.set_label("Attention Intensity")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
        
        # Use self.log_dir instead of self.attention_log_dir
        filename = os.path.join(self.log_dir, f"attention_epoch_{epoch+1}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved attention map: {filename}")
        
    def set_model(self, model):
        super().set_model(model)
    def set_encoder(self, encoder):
        self.encoder = encoder
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_every_n_epochs != 0:
            return

        try:
            print(f"\n[AttentionVisualization] Generating attention map for epoch {epoch+1}...")
            random_index = random.randint(0, len(self.validation_data) - 1)
            sattelite_image, drone_image = self.validation_data[random_index]
            
            # Generate attention visualization if requested
            print("[CALLBACK] Generating attention maps...")
            self.visualize_attention(sattelite_image, drone_image, epoch)
                
        except Exception as e:
            print(f"[CALLBACK] Error in on_epoch_end: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def visualize_attention(self, anchor, positive, epoch):
        tokens = self.encoder(positive)
        outputs = self.model(tokens, training=True)
        _, attn_scores_pos = outputs
        
        attn_scores_pos = feature_norm(attn_scores_pos).numpy()
        
        attn_map_pos = np.mean(attn_scores_pos[0], axis=(0, 1))
        
        num_tokens = attn_map_pos.shape[0]
        H = W = int(np.sqrt(num_tokens))
        if H * W != num_tokens:
            raise ValueError(f"Cannot reshape {num_tokens} tokens into square. H={H}, W={W}")

        attn_map_pos = attn_map_pos.reshape(H, W)

        attn_map_pos_resized = cv2.resize(attn_map_pos, (anchor.shape[1], anchor.shape[0]))
        
        self._save_attention_visualization(anchor, positive, 
                                        attn_map_pos_resized, epoch)

    