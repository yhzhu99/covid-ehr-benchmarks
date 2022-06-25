import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        demo_dim,
        lab_dim,
        max_visits,
        hidden_dim,
        num_layers,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(Transformer, self).__init__()

        # hyperparameters
        self.demo_dim = demo_dim
        self.lab_dim = lab_dim
        self.max_visits = max_visits
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(demo_dim + lab_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(max_visits)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=4 * hidden_dim,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=2
        )

    def forward(self, x):
        batch_size, time_steps, _ = x.size()
        x = self.proj(x)
        out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        for cur_time in range(time_steps):
            cur_x = x[:, : cur_time + 1, :]
            cur_x = self.transformer_encoder(cur_x)
            out[:, cur_time, :] = torch.mean(cur_x, dim=1)
        return out
