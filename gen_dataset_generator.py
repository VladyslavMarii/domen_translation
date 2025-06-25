from pathlib import Path
import sys
import h5py
import numpy as np
from tf_keras.utils import Sequence
# Add parent directory to Python path to find config.py
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from config import BATCH_SIZE, IMAGE_CHANNELS, IMAGE_RESOLUTION, TRAIN_DATASET_FILE_PATH, VAL_DATASET_FILE_PATH


class HDF5DataGenerator(Sequence):
    def __init__(self, file_path, batch_size, image_shape, shuffle=True):
        """
        Custom Sequence data generator for HDF5 files.
        
        Args:
            file_path (str): Path to the HDF5 file.
            batch_size (int): Number of samples per batch.
            image_shape (tuple): Shape of the input images (height, width, channels).
            shuffle (bool): Whether to shuffle the data after each epoch.
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.shuffle = shuffle
        train_path = Path(TRAIN_DATASET_FILE_PATH)
        val_path = Path(VAL_DATASET_FILE_PATH)

        print(f"Train dataset exists: {train_path.exists()}")
        print(f"Train dataset absolute path: {train_path.absolute()}")
        print(f"Val dataset exists: {val_path.exists()}")
        print(f"Val dataset absolute path: {val_path.absolute()}")
        # Open the HDF5 file
        with h5py.File(self.file_path, 'r') as hf:
            self.num_samples = len(hf['anchor'])
        
        # Initialize indices and shuffle
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generates one batch of data.
        """
        # Calculate start and end indices for the batch
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Initialize arrays to hold the batch data
        batch_anchors = np.zeros((len(batch_indices), *self.image_shape), dtype=np.float32)
        batch_positives = np.zeros((len(batch_indices), *self.image_shape), dtype=np.float32)

        
        # Load the batch data from the HDF5 file
        with h5py.File(self.file_path, 'r') as hf:
            for i, idx in enumerate(batch_indices):
                batch_anchors[i] = hf['anchor'][idx]
                batch_positives[i] = hf['positive'][idx]
        
        return batch_anchors, batch_positives
    
    def on_epoch_end(self):
        """
        Shuffles the data after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_datasets(train_dataset_path, val_dataset_path):
    """
    Creates training and validation datasets using the custom Sequence generator.
    """
    image_shape = (IMAGE_RESOLUTION["height"], IMAGE_RESOLUTION["width"], IMAGE_CHANNELS)
    
    train_generator = HDF5DataGenerator(
        file_path=train_dataset_path,
        batch_size=BATCH_SIZE,
        image_shape=image_shape,
        shuffle=False
    )

    val_generator = HDF5DataGenerator(
        file_path=val_dataset_path,
        batch_size=BATCH_SIZE,
        image_shape=image_shape,
        shuffle=False
    )
    
    return train_generator, val_generator
