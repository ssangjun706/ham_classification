import torch.nn as nn
import torch


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, h_dim):
        super().__init__()
        self.seq_len = seq_len
        self.h_dim = h_dim

        _seq = torch.arange(self.seq_len).reshape((-1, 1))
        _seq = _seq.expand((self.seq_len, self.h_dim))

        _freq = 2 * torch.floor(torch.arange(self.seq_len) * 0.5) / self.h_dim
        _freq = torch.pow(10000, (_freq)).reshape((-1, 1))
        _freq = _freq.expand((self.seq_len, self.h_dim))

        _pe = _seq / _freq
        _pe[:, ::2] = torch.sin(_pe[:, ::2])
        _pe[:, 1::2] = torch.cos(_pe[:, 1::2])
        self.encoding = _pe

    def forward(self, x):
        encoding = self.encoding.expand((x.shape[0], self.seq_len, self.h_dim))
        encoding = encoding.to(x.device)
        return x + encoding


class ViT(nn.Module):
    def __init__(self, seq_len, in_features, h_dim, num_layers=12, nhead=8):
        super().__init__()
        self.seq_len = seq_len + 1
        self.in_features = in_features
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.token = torch.randn((1, self.h_dim))
        self.projection = nn.Sequential(
            nn.Linear(self.in_features, self.h_dim),
            nn.ReLU(),
        )

        self.pe = PositionalEmbedding(self.seq_len, self.h_dim)

        _encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.h_dim,
            nhead=self.nhead,
        )
        self.transformer = nn.TransformerEncoder(
            _encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.h_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.h_dim, 3072),
            nn.ReLU(),
            nn.Linear(3072, 7),
            nn.ReLU(),
        )

    def forward(self, x):
        _size = (x.shape[0], 1, self.h_dim)
        self.token = self.token.expand(_size).to(x.device)
        x = self.projection(x)
        x = torch.concat((self.token, x), dim=1)
        x = self.pe(x)
        out = self.transformer(x)
        pred = self.mlp(out[:, 0, :])
        return pred
