import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # --- Encoder (Compress spatial info) ---
        self.encoder = nn.Sequential(
            # Input: (1, 32, 32)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 8, 8)
            nn.ReLU()
        )
        
        # --- Decoder (Reconstruct clean hotspot) ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (1, 32, 32)
            nn.Sigmoid() # Force output between 0 and 1 (Heatmap)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded