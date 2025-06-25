import glob
import os
import random
import h5py
import numpy as np
import tensorflow as tf
from config import BATCH_SIZE, VAL_DATASET_FILE_PATH
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

# Create zoom layer at module level
ZOOM_LAYER = tf.keras.layers.RandomZoom(
    height_factor=(-0.3, -0.2),  # Zoom in by 0-30%
    width_factor=(-0.3, -0.2),   # Same zoom factor for width to maintain aspect ratio
    fill_mode='nearest'
)

def zoom_and_resize(image, target_size=(400, 400)):
    """
    Apply random zoom to image and resize to target size.
    
    Args:
        image: Input image tensor
        target_size: Desired output size (height, width)
    """
    if len(image.shape) == 4:  # Batched input
        return tf.map_fn(
            lambda img: zoom_and_resize(img, target_size),
            image,
            fn_output_signature=tf.float32,
        )
    
    # Add batch dimension for zoom layer
    needs_batch_dim = len(image.shape) == 3
    if needs_batch_dim:
        image = tf.expand_dims(image, 0)
    
    # Apply random zoom
    zoomed = ZOOM_LAYER(image, training=True)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        zoomed = tf.squeeze(zoomed, 0)
    
    # Ensure output size is correct
    resized = tf.image.resize(zoomed, target_size)
    
    return resized

def crop_and_pad(image, crop_size=(280, 280), target_size=(400, 400)):
    if len(image.shape) == 4:  # Batched input
        return tf.map_fn(
            lambda img: crop_and_pad(img, crop_size, target_size),
            image,
            fn_output_signature=tf.float32,
        )
    
    # Random crop and pad for training
    cropped_image = tf.image.random_crop(image, size=(crop_size[0], crop_size[1], image.shape[-1]))
    pad_height = target_size[0] - crop_size[0]
    pad_width = target_size[1] - crop_size[1]
    padded_image = tf.image.pad_to_bounding_box(
        cropped_image,
        offset_height=pad_height//2,
        offset_width=pad_width//2,
        target_height=target_size[0],
        target_width=target_size[1],
    )
    return padded_image

rotation_layer = tf.keras.layers.RandomRotation(factor=0.0555555)  # ±20 degrees

def preprocess_with_cropping(anchor, positive, negative):
    """Applies augmentation to all images in a triplet"""
    # Apply random rotation to each image
    anchor = rotation_layer(anchor)
    positive = rotation_layer(positive)
    negative = rotation_layer(negative)
    
    # Crop and pad each image
    anchor = zoom_and_resize(anchor)
    positive = zoom_and_resize(positive)
    negative = zoom_and_resize(negative)
    
   
    return anchor, positive, negative

def feature_norm(x):
        """Min-Max Normalization that ensures safe division and prevents NaNs."""
        x_min = tf.reduce_min(x, axis=-1, keepdims=True)
        x_max = tf.reduce_max(x, axis=-1, keepdims=True)

        # Prevent division by zero by checking if x_max == x_min
        range_x = x_max - x_min
        safe_range = tf.where(range_x > 1e-6, range_x, tf.ones_like(range_x))  # Avoid zero division

        # Apply Min-Max Normalization safely
        x = (x - x_min) / safe_range  # Now the denominator is always > 0

        # Replace NaNs with zeros
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

        return x


def get_random_batch_from_h5(file_path, just_one_sample=False):
    """
    Opens the HDF5 file, picks `batch_size` random indices, and returns
    a batch of (anchor, positive, negative) as tf.Tensors.
    """
    with h5py.File(file_path, 'r') as hf:
        total_samples = len(hf['anchor'])  # Adjust to your structure if needed
        # pick random unique indices
        idxs = random.sample(range(total_samples), min(BATCH_SIZE, total_samples))
        
        anchors = []
        positives = []
       
        for i in idxs:
            anchors.append(hf['anchor'][i])
            positives.append(hf['positive'][i])
       

    # Convert to tf.Tensor. 
    # If your HDF5 data is stored as float32 [0..1], you might omit casting, or adjust as needed.
    anchor_batch = tf.convert_to_tensor(anchors, dtype=tf.float32)     # shape: (batch_size, 400, 400, 3)
    positive_batch = tf.convert_to_tensor(positives, dtype=tf.float32) # shape: (batch_size, 400, 400, 3)
    

    return (anchor_batch, positive_batch) if not just_one_sample else (anchor_batch[0], positive_batch[0])