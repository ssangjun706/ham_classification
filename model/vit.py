import torch.nn as nn
import torch


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, h_dim):
        super().__init__()
        self.seq_len = seq_len
        self.h_dim = h_dim

        assert seq_len % 2 == 0, "Sequence length should be divided into 2."
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
    def __init__(
        self,
        image_size,
        patch_size,
        h_dim,
        mlp_dim,
        num_classes,
        num_layers=12,
        nhead=12,
    ):
        super().__init__()
        self.h_dim = h_dim
        seq_len = (image_size // patch_size) ** 2
        in_features = (patch_size**2) * 3

        self.cls_token = nn.Parameter(torch.randn((1, h_dim)))
        self.to_patch = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.embed = nn.Sequential(
            nn.Linear(in_features, h_dim), nn.BatchNorm2d(h_dim), nn.ReLU()
        )

        self.pe = nn.Sequential(
            PositionalEmbedding(seq_len + 1, h_dim),
            nn.Dropout(0.1),
        )

        _encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=nhead,
            dim_feedforward=mlp_dim,
        )

        self.transformer = nn.TransformerEncoder(
            _encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(h_dim),
        )

        self.mlp = nn.Linear(h_dim, num_classes)

    def forward(self, x):
        cls_token = self.cls_token.expand((x.shape[0], 1, self.h_dim))
        x = self.to_patch(x).permute(0, 2, 1)
        x = torch.concat((cls_token, self.embed(x)), dim=1)
        x = self.pe(x)

        out = self.transformer(x)
        pred = self.mlp(out[:, 0, :])
        return pred
