from torch import nn


class GRU(nn.Module):
    def __init__(
        self,
        demo_dim,
        lab_dim,
        max_visits,
        hidden_dim,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(GRU, self).__init__()

        # hyperparameters
        self.demo_dim = demo_dim
        self.lab_dim = lab_dim
        self.max_visits = max_visits
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(demo_dim + lab_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(max_visits)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x = self.proj(x)
        # x = self.act(x)
        # x = self.bn(x)
        x, _ = self.gru(x)
        return x
