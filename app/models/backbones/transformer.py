from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        demo_dim,
        lab_dim,
        max_visits,
        hidden_dim,
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
            d_model=hidden_dim, nhead=4, dim_feedforward=512, activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=1
        )

    def forward(self, x):
        x = self.proj(x)
        # x = self.act(x)
        # x = self.bn(x)

        x = self.transformer_encoder(x)
        return x
