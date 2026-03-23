import torch
import torch.nn as nn


class ColorDecoder(nn.Module):
    """
    Small MLP that decodes per-Gaussian color embeddings into RGB values.
    
    Input: [N, color_embed_dim] color embeddings
    Output: [N, 3] RGB colors in [0, 1]
    
    Extensible if output is single bands or multi-spectral channels.
    """
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=3, num_hidden_layers=2):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Output in [0, 1] for RGB
        ])
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
