import torch
import torch.nn as nn
from blocks import ResnetBlock, Downsample, Upsample, Normalize, nonlinearity

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, downsample_steps, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=1, padding=2)])
        for _ in range(downsample_steps):
            self.layers.append(nn.Sequential(ResnetBlock(hidden_channels, hidden_channels, dropout), Downsample(hidden_channels)))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, upsample_steps, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(upsample_steps):
            self.layers.append(nn.Sequential(ResnetBlock(hidden_channels, hidden_channels, dropout), Upsample(hidden_channels)))
        self.norm = Normalize(hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return torch.tanh(x)