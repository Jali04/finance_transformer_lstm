# src/model.py
import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, in_features, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: [B, S, F]
        h = self.input_proj(x)
        h = h + self.pos_embed[:, :h.size(1), :]
        h = self.transformer(h)
        
        # Global Average Pooling über die Zeitdimension (S)
        pooled = h.mean(dim=1) 
        return self.reg_head(pooled)