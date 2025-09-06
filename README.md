# Drone Path Planning in a High-Density City: A CNN-Transformer based Model for Seoul

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

# TransPath-with-Risk-Map
<div align="center">
<img width="263" height="251" alt="A star Algorithm visualization" src="https://github.com/user-attachments/assets/0c06ed0c-1995-44cf-9fc6-d072212f8841" />
<p>A* Algorithm</p>
</div>
<br>
<div align="center">
<img width="263" height="264" alt="Predicted Path Probability Map" src="https://github.com/user-attachments/assets/ce15a10b-f60e-432a-823e-30b77b91dcf5" />
<p>Predicted PPM</p>
</div>
<br>
<div align="center">
<img width="263" height="263" alt="Focal Search with PPM visualization" src="https://github.com/user-attachments/assets/7b10e8ed-e89d-4465-b8c4-682edfefb18b" />
<p>Focal Search with PPM</p>
</div>
