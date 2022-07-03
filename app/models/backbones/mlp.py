from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        demo_dim,
        lab_dim,
        max_visits,
        hidden_dim,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(MLP, self).__init__()

        # hyperparameters
        self.demo_dim = demo_dim
        self.lab_dim = lab_dim
        self.max_visits = max_visits
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(demo_dim + lab_dim, hidden_dim)
        self.act = act_layer()
        self.bn = nn.BatchNorm1d(max_visits)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            self.act,
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x, device, info=None):
        """extra info is not used here"""
        x = self.proj(x)
        # x = self.act(x)
        # x = self.bn(x)
        x = self.mlp(x)
        return x
