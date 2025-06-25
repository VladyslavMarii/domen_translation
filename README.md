# Drone-to-Satellite Image Translation with Transformers

A transformer-based model that translates drone imagery to satellite-like visual representations while maintaining the same viewing perspective. This project explores cross-domain visual translation using attention mechanisms.

## Project Overview

This experiment teaches a neural network to transform drone camera imagery to match the visual characteristics of satellite imagery from the same viewing angle - not changing perspective, but adapting visual style, lighting, and texture patterns.

**System Components:**
- **DroneEncoder**: Transformer model for token translation
- **Satellite Encoder/Decoder**: Image patch processing 
- **Training Pipeline**: End-to-end training with monitoring callbacks
- **Visualization Tools**: Attention maps and reconstruction analysis

## Results

![Training Results](results.png)
*Training progression showing representation collapse and recovery phases*

The model successfully transforms drone imagery visual characteristics while preserving spatial relationships:
- Accurate waterway and road structure preservation
- Effective visual style transfer to satellite appearance
- Maintained geometric layouts and terrain organization

## Architecture

**Approach:** Transformer-based architecture operating on image patches ("tokens")
1. Images divided into small patches 
2. Token-based transformation using multi-head attention
3. Context-aware processing through self-attention mechanisms

**Key Components:**
- **DroneEncoder**: Multi-head attention transformer block
- **Satellite Encoder/Decoder**: Patch processing and reconstruction
- **Training Callbacks**: Attention visualization, token matching, model checkpointing

**Loss Function:** MSE between predicted and real satellite tokens in token space


## Installation

```bash
git clone <repository-url>
cd transformers
pip install -r requirements.txt
```

**System Requirements:**
- Intel i7 11th Gen, 96GB RAM, 2TB M.2 SSD, RTX 3090
- GPU acceleration recommended for transformer computations

**Data Structure:**
```
data/
├── datasets/
│   ├── train_dataset.tfrecord
│   └── val_dataset.tfrecord
└── models/
```

## Configuration

**Main Config (`config.py`):** Training ID, image resolution (400x400), dataset paths, batch size, epochs
**Model Config (`raw_gen_model/raw_gen_config.py`):** Patch size (40x40), attention heads (8)

## Usage

1. **Setup dataset:** Paired drone/satellite images in TFRecord format
2. **Configure paths:** Update `config.py` with dataset locations  
3. **Run training:** Execute `jupyter notebook raw_gen_train.ipynb`

**Training Process:**
- Encodes drone images → transforms tokens → compares with satellite tokens
- MSE loss optimization, Adam optimizer (1e-4 learning rate)
- Automatic model checkpointing and visualization generation

**Monitoring:** Attention maps, token matching metrics, reconstruction visualization in `logs/{TRAINING_ID}/`

## File Structure

```
├── config.py                              # Main configuration
├── helper.py                              # Utility functions  
├── gen_dataset_generator.py               # Dataset utilities
├── raw_gen_train.ipynb                    # Training notebook
├── requirements.txt                       # Dependencies
└── raw_gen_model/
    ├── raw_gen_config.py                  # Model config
    ├── raw_gen_helper.py                  # Encoder/decoder
    ├── raw_gen_encoders/
    │   └── raw_gen_drone_encoder.py       # Transformer model
    └── raw_gen_callbacks/                 # Training callbacks
```

## Troubleshooting

**Common Issues:**
- **GPU Memory:** Reduce batch size or enable memory growth
- **Dataset Format:** Ensure TFRecord files with paired (satellite, drone) data
- **Dependencies:** Install `tf-keras`, `opencv-python`, and other requirements

**Performance:** Use GPU acceleration, monitor memory usage, adjust patch size/attention heads based on hardware

## Key Insights

**Training Journey:** The model experienced an unexpected "representation collapse" phase (producing black images) before recovering with improved feature recognition - demonstrating the importance of patience in deep learning experiments.

**Lessons Learned:**
- Well-matched image pairs are crucial for domain translation
- Transformer architectures excel at cross-domain visual translation
- Training patience pays off - apparent failures can precede breakthroughs

**Future Directions:** Multi-scale approaches, stronger regularization, diverse terrain testing, hybrid architectures
