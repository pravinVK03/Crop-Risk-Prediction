import torch
from torch import nn

from .config import ModelConfig


class NumericalTokenizer(nn.Module):
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabTransformerClassifier(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        num_features: int,
        num_classes: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, config.d_model) for cardinality in cat_cardinalities]
        )
        self.num_tokenizer = NumericalTokenizer(num_features=num_features, d_model=config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.embedding_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.ff_multiplier,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=config.n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, num_classes),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        cat_tokens = [embedding(x_cat[:, index]) for index, embedding in enumerate(self.cat_embeddings)]
        cat_tokens = torch.stack(cat_tokens, dim=1)
        num_tokens = self.num_tokenizer(x_num)

        batch_size = x_cat.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, cat_tokens, num_tokens), dim=1)
        tokens = self.embedding_dropout(tokens)

        encoded = self.encoder(tokens)
        cls_representation = encoded[:, 0, :]
        return self.classifier(cls_representation)
