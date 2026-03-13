import torch
import torch.nn as nn


class ColorDecoder(nn.Module):
    """
    Small MLP that decodes per-Gaussian color embeddings into RGB values.
    
    Input: [N, color_embed_dim] color embeddings
    Output: [N, 3] RGB colors in [0, 1]
    
    Architecture designed to be extensible for Phase 2+ where the output
    can change to single bands or multi-spectral channels.
    """
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Output in [0, 1] for RGB
        )
    
    def forward(self, x):
        return self.net(x)
