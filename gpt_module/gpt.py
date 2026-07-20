import torch
import torch.nn as nn

# Import already prepared `TransformerBlock` abstraction
import sys 
sys.path.append("../..")
from transformer_module import TransformerBlock
from transformer_module import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) ## projecting vocabulary to embeddings dimensional space.
        self.pos_emb = nn.Embedding(cfg["context_window_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) ## projecting embeddings to vocabulary_size dimensional space.

    def forward(self, in_idx):
        batch, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        ## represents the scores for all tokens in the vocabulary, higher score implies higher chances of being next token.
        logits = self.out_head(x)
        
        return logits