import torch
import torch.nn as nn
from encoder_decoder import Encoder, Decoder
from blocks import PosEmbeds
from transformers import SpatialTransformer

class TransPathPPM(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, hidden_channels=64, attn_blocks=4, attn_heads=4, 
                 cnn_dropout=0.15, attn_dropout=0.15, downsample_steps=3, resolution=(64, 64)):
        super().__init__()
        heads_dim = hidden_channels // attn_heads
        self.encoder = Encoder(in_channels, hidden_channels, downsample_steps, cnn_dropout)
        self.pos = PosEmbeds(hidden_channels, (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps))
        self.transformer = SpatialTransformer(hidden_channels, attn_heads, heads_dim, attn_blocks, attn_dropout)
        self.decoder_pos = PosEmbeds(hidden_channels, (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps))
        self.decoder = Decoder(hidden_channels, out_channels, downsample_steps, cnn_dropout)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.pos(x)
        x = self.transformer(x)
        x = self.decoder_pos(x)
        x = self.decoder(x)
        return x