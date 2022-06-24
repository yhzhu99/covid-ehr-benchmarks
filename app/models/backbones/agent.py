import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch import nn
from torch.autograd import Variable
from torch.utils import data

RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True


class Agent(nn.Module):
    def __init__(
        self,
        cell="gru",
        use_baseline=True,
        n_actions=10,
        n_units=64,
        fusion_dim=128,
        lab_dim=73,
        n_hidden=128,
        demo_dim=2,
        # n_output=1,
        dropout=0.0,
        lamda=0.5,
    ):
        super(Agent, self).__init__()

        self.cell = cell
        self.use_baseline = use_baseline
        self.n_actions = n_actions
        self.n_units = n_units
        self.lab_dim = lab_dim
        self.n_hidden = n_hidden
        # self.n_output = n_output
        self.dropout = dropout
        self.lamda = lamda
        self.fusion_dim = fusion_dim
        self.demo_dim = demo_dim

        self.agent1_action = []
        self.agent1_prob = []
        self.agent1_entropy = []
        self.agent1_baseline = []
        self.agent2_action = []
        self.agent2_prob = []
        self.agent2_entropy = []
        self.agent2_baseline = []

        self.agent1_fc1 = nn.Linear(self.n_hidden + self.demo_dim, self.n_units)
        self.agent2_fc1 = nn.Linear(self.lab_dim + self.demo_dim, self.n_units)
        self.agent1_fc2 = nn.Linear(self.n_units, self.n_actions)
        self.agent2_fc2 = nn.Linear(self.n_units, self.n_actions)
        if use_baseline == True:
            self.agent1_value = nn.Linear(self.n_units, 1)
            self.agent2_value = nn.Linear(self.n_units, 1)

        if self.cell == "lstm":
            self.rnn = nn.LSTMCell(self.lab_dim, self.n_hidden)
        else:
            self.rnn = nn.GRUCell(self.lab_dim, self.n_hidden)

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        if dropout > 0.0:
            self.nn_dropout = nn.Dropout(p=dropout)
        self.init_h = nn.Linear(self.demo_dim, self.n_hidden)
        self.init_c = nn.Linear(self.demo_dim, self.n_hidden)
        self.fusion = nn.Linear(self.n_hidden + self.demo_dim, self.fusion_dim)
        # self.output = nn.Linear(self.fusion_dim, self.n_output)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def choose_action(self, observation, agent=1):
        observation = observation.detach()

        if agent == 1:
            result_fc1 = self.agent1_fc1(observation)
            result_fc1 = self.tanh(result_fc1)
            result_fc2 = self.agent1_fc2(result_fc1)
            if self.use_baseline == True:
                result_value = self.agent1_value(result_fc1)
                self.agent1_baseline.append(result_value)
        else:
            result_fc1 = self.agent2_fc1(observation)
            result_fc1 = self.tanh(result_fc1)
            result_fc2 = self.agent2_fc2(result_fc1)
            if self.use_baseline == True:
                result_value = self.agent2_value(result_fc1)
                self.agent2_baseline.append(result_value)

        probs = self.softmax(result_fc2)
        m = torch.distributions.Categorical(probs)
        actions = m.sample()

        if agent == 1:
            self.agent1_entropy.append(m.entropy())
            self.agent1_action.append(actions.unsqueeze(-1))
            self.agent1_prob.append(m.log_prob(actions))
        else:
            self.agent2_entropy.append(m.entropy())
            self.agent2_action.append(actions.unsqueeze(-1))
            self.agent2_prob.append(m.log_prob(actions))

        return actions.unsqueeze(-1)

    def forward(self, x):
        demo = x[:, :, : self.demo_dim]
        labtest = x[:, :, self.demo_dim :]

        batch_size = labtest.size(0)
        time_step = labtest.size(1)
        feature_dim = labtest.size(2)
        assert feature_dim == self.lab_dim

        self.agent1_action = []
        self.agent1_prob = []
        self.agent1_entropy = []
        self.agent1_baseline = []
        self.agent2_action = []
        self.agent2_prob = []
        self.agent2_entropy = []
        self.agent2_baseline = []

        cur_h = self.init_h(demo)
        if self.cell == "lstm":
            cur_c = self.init_c(demo)

        for cur_time in range(time_step):
            cur_input = labtest[:, cur_time, :]

            if cur_time == 0:
                obs_1 = cur_h
                obs_2 = cur_input
                obs_1 = torch.cat((obs_1, demo), dim=1)
                obs_2 = torch.cat((obs_2, demo), dim=1)
                self.choose_action(obs_1, 1).long()
                self.choose_action(obs_2, 2).long()

                observed_h = (
                    torch.zeros_like(cur_h, dtype=torch.float32)
                    .view(-1)
                    .repeat(self.n_actions)
                    .view(self.n_actions, batch_size, self.n_hidden)
                )
                action_h = cur_h
                if self.cell == "lstm":
                    observed_c = (
                        torch.zeros_like(cur_c, dtype=torch.float32)
                        .view(-1)
                        .repeat(self.n_actions)
                        .view(self.n_actions, batch_size, self.n_hidden)
                    )
                    action_c = cur_c

            else:
                observed_h = torch.cat((observed_h[1:], cur_h.unsqueeze(0)), 0)

                obs_1 = observed_h.mean(dim=0)
                obs_2 = cur_input
                obs_1 = torch.cat((obs_1, demo), dim=1)
                obs_2 = torch.cat((obs_2, demo), dim=1)
                act_idx1 = self.choose_action(obs_1, 1).long()
                act_idx2 = self.choose_action(obs_2, 2).long()
                batch_idx = torch.arange(batch_size, dtype=torch.long).unsqueeze(-1)
                action_h1 = observed_h[act_idx1, batch_idx, :].squeeze(1)
                action_h2 = observed_h[act_idx2, batch_idx, :].squeeze(1)
                action_h = (action_h1 + action_h2) / 2
                if self.cell == "lstm":
                    observed_c = torch.cat((observed_c[1:], cur_c.unsqueeze(0)), 0)
                    action_c1 = observed_c[act_idx1, batch_idx, :].squeeze(1)
                    action_c2 = observed_c[act_idx2, batch_idx, :].squeeze(1)
                    action_c = (action_c1 + action_c2) / 2

            if self.cell == "lstm":
                weighted_h = self.lamda * action_h + (1 - self.lamda) * cur_h
                weighted_c = self.lamda * action_c + (1 - self.lamda) * cur_c
                rnn_state = (weighted_h, weighted_c)
                cur_h, cur_c = self.rnn(cur_input, rnn_state)
            else:
                weighted_h = self.lamda * action_h + (1 - self.lamda) * cur_h
                cur_h = self.rnn(cur_input, weighted_h)

        if self.dropout > 0.0:
            cur_h = self.nn_dropout(cur_h)
        cur_h = torch.cat((cur_h, demo), dim=1)
        cur_h = self.fusion(cur_h)
        cur_h = self.relu(cur_h)
        # output = self.output(cur_h)
        # output = self.sigmoid(output)

        return cur_h
