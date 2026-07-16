import torch.nn as nn

from .feedforward import FeedForward
from .layer_norm import LayerNorm

# Import already prepared `MultiHeadAttention` abstraction
import sys 
sys.path.append("../..")
from attention_mechanism_module import MultiHeadAttention 

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_window_length=cfg["context_window_size"],
            num_heads=cfg["n_heads"], 
            dropout_rate=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        self.ff = FeedForward(cfg)
        
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x