# projected_unimodal_tf.py
import torch
import torch.nn as nn
from modules import Unimodal_TF

class ProjectedUnimodalTF(nn.Module):
    def __init__(self, input_dim=207, transformer_d_model=256, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.encoder = Unimodal_TF(
            input_dim=input_dim, 
            transformer_d_model=transformer_d_model,
            num_heads_=num_heads,
            num_layers_=num_layers,
            dropout=dropout,
            return_all_tokens=False
        )

    def forward(self, x):
        
        return self.encoder(x) 
