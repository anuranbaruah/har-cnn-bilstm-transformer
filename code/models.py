"""Model definitions. Input to every model: (batch, 128 timesteps, 9 channels)."""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Member 1 (Anuran Baruah): 3 x [Conv1D, BatchNorm, ReLU, MaxPool] -> global average pool -> linear."""
    def __init__(self, in_ch=9, n_classes=6, ch=(64, 128, 128), k=5, dropout=0.3):
        super().__init__()
        layers, c_in = [], in_ch
        for c in ch:
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x):                      # x: (B, T, C)
        h = self.features(x.transpose(1, 2))   # (B, C', T')
        return self.head(h.mean(dim=-1))


class BiLSTMAttn(nn.Module):
    """Member 2 (Syed Ayaan Shere): 2-layer bidirectional LSTM -> additive attention pooling -> linear."""
    def __init__(self, in_ch=9, n_classes=6, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(2 * hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):                      # x: (B, T, C)
        h, _ = self.lstm(x)                    # (B, T, 2H)
        w = torch.softmax(self.attn(h), dim=1) # (B, T, 1)
        return self.head((w * h).sum(dim=1))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        import math
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                      # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class TransformerHAR(nn.Module):
    """Member 3 (Raghav Nagpal): linear channel embedding -> sinusoidal positions ->
    Transformer encoder blocks -> mean pooling over time -> linear."""
    def __init__(self, in_ch=9, n_classes=6, d_model=64, heads=4, layers=3, ff=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_ch, d_model)
        self.pos = PositionalEncoding(d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.3), nn.Linear(d_model, n_classes))

    def forward(self, x):                      # x: (B, T, C)
        h = self.encoder(self.pos(self.embed(x)))
        return self.head(h.mean(dim=1))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
