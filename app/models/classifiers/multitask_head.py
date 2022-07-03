from torch import nn
import torch


class MultitaskHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(MultitaskHead, self).__init__()
        self.hidden_dim = (hidden_dim,)
        self.output_dim = (output_dim,)
        self.act = act_layer()
        self.prediction_head_outcome = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(drop),
            nn.Sigmoid(),
        )

        self.prediction_head_los = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(drop),
        )

    def forward(self, x, device):
        # x = self.act(x)
        outcome = self.prediction_head_outcome(x.to(device=device))
        los = self.prediction_head_los(x.to(device=device))
        return outcome, los
