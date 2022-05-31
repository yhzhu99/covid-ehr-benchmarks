from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        lab_dim,
        demo_dim,
        max_visits,
        hidden_dim,
        output_dim,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(Transformer, self).__init__()

        # hyperparameters
        self.lab_dim = lab_dim
        self.demo_dim = demo_dim
        self.max_visits = max_visits
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.proj = nn.Linear(demo_dim + lab_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(max_visits)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=512, activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=1
        )

        self.act = act_layer()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x)
        # x = self.act(x)
        # x = self.bn(x)

        # x, _ = self.gru(x)
        x = self.transformer_encoder(x)

        x = self.drop(x)
        x = self.fc(x)
        x = self.drop(x)

        # x = self.sigmoid(x)
        return x
