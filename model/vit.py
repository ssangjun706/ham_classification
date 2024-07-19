import torch
from torch import nn

from model.transformer import Transformer


class Reshape(nn.Module):
    def __init__(self, reshape):
        super().__init__()
        self.reshape = reshape

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view((batch_size, *self.reshape))


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

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size),
            Reshape((seq_len, in_features)),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
        )

        self.cls_token = nn.Parameter(torch.randn((1, 1, h_dim)))
        self.pe = PositionalEmbedding(seq_len + 1, h_dim)

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=h_dim,
        #     nhead=nhead,
        #     dim_feedforward=mlp_dim,
        # )

        # self.transformer = nn.TransformerEncoder(
        #     encoder_layer=encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(h_dim)
        # )
        self.transformer = Transformer(h_dim, num_layers, nhead, 64, mlp_dim)
        self.mlp = nn.Linear(h_dim, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand((x.shape[0], 1, self.h_dim))
        x = torch.cat((cls_token, x), dim=1)
        x = self.pe(x)
        x = self.transformer(x)
        return self.mlp(x[:, 0])
