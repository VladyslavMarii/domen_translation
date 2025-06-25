from pathlib import Path
import tensorflow as tf

#PATH CONFIG
TRAINING_ID = "model_training_001"

PROJECT_ROOT = Path(__file__).parent
DATASETS_PATH = PROJECT_ROOT / "data/datasets"
MODELS_PATH = PROJECT_ROOT / "data/models"
LOGS_DIR = PROJECT_ROOT / "logs" / TRAINING_ID

#MODEL CONFIG
IMAGE_RESOLUTION = {
    "height": 400,
    "width": 400
}
IMAGE_CHANNELS = 3
ANNOTATION_LABELS = 1

# GEN MODEL CONFIG (formerly from gen_config.py)
BATCH_SIZE = 16
CONV_FILTERS = [32, 64, 128]
EPOCHS = 5000

# GEN MODEL PATHS
STORAGE_PATH = DATASETS_PATH / '5000_storage.h5'
TRAIN_DATASET_FILE_PATH = DATASETS_PATH / "paired_train_dataset.h5"
VAL_DATASET_FILE_PATH = DATASETS_PATH / "paired_val_dataset.h5"
TEST_DATASET_FILE_PATH = DATASETS_PATH / "paired_test_dataset.h5"
CNN_MODEL_PATH = MODELS_PATH / 'pretrained_large_dataset_feature.keras'

# GEN MODEL PARAMETERS
ISLAND_THRESHOLD = 0.25
FEATURE_WEIGHT = 0
NUM_HEADS = 8

# Formula: token_loss = pos_token_loss * POS_TOKEN_WEIGHT + neg_token_loss * (1 - POS_TOKEN_WEIGHT)
BASE_TOKEN_SIZE = 4 + 1024 + CONV_FILTERS[1] * 2
# 4 x, y, width, height - of the box
# 1024 Flattened resized bounding box (cause all boxes are resized to 32x32)
# CONV_FILTERS[2] - Mean and Max activations per filter
DIM = 786

GENERATOR_OUTPUT_SIGNATURE = (
    {
        'anchor': tf.TensorSpec(shape=(IMAGE_RESOLUTION["height"], IMAGE_RESOLUTION["width"], IMAGE_CHANNELS), dtype=tf.float32),
        'positive': tf.TensorSpec(shape=(IMAGE_RESOLUTION["height"], IMAGE_RESOLUTION["width"], IMAGE_CHANNELS), dtype=tf.float32),
        'negative': tf.TensorSpec(shape=(IMAGE_RESOLUTION["height"], IMAGE_RESOLUTION["width"], IMAGE_CHANNELS), dtype=tf.float32)
    }
)
