# Deep Learning Project

A PyTorch implementation of TransPathPPM, a deep learning model combining CNN encoders/decoders with spatial transformer attention mechanisms.

## Overview

This project implements a neural network architecture that uses:
- CNN-based encoder-decoder structure
- Spatial transformer attention blocks
- Position embeddings
- Configurable attention heads and blocks

## Files

- `model.py` - Main TransPathPPM model implementation
- `encoder_decoder.py` - Encoder and Decoder implementations
- `blocks.py` - Position embeddings and building blocks
- `transformers.py` - Spatial transformer implementation
- `attention.py` - Attention mechanisms
- `train.py` - Training script
- `inference.py` - Inference utilities
- `dataset.py` - Dataset handling
- `training_utils.py` - Training utilities

## Requirements

- PyTorch
- NumPy
- Other dependencies as needed

## Usage

```python
from model import TransPathPPM

# Initialize model
model = TransPathPPM(
    in_channels=2,
    out_channels=1,
    hidden_channels=64,
    attn_blocks=4,
    attn_heads=4
)

# Forward pass
output = model(input_tensor)
```

## Training

Run the training script:
```bash
python train.py
```

## License

[Add your license here]