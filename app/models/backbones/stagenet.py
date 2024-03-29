import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch import nn
from torch.autograd import Variable
from torch.utils import data


class StageNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        conv_size,
        # output_dim,
        levels,
        dropconnect=0.5,
        dropout=0.5,
        dropres=0.3,
    ):
        super(StageNet, self).__init__()

        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim
        self.conv_size = conv_size
        # self.output_dim = output_dim
        self.levels = levels
        self.chunk_size = hidden_dim // levels

        self.kernel = nn.Linear(int(input_dim + 1), int(hidden_dim * 4 + levels * 2))
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(
            int(hidden_dim + 1), int(hidden_dim * 4 + levels * 2)
        )
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(int(hidden_dim), int(hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(hidden_dim // 6), int(hidden_dim))
        self.nn_conv = nn.Conv1d(int(hidden_dim), int(self.conv_dim), int(conv_size), 1)
        # self.nn_output = nn.Linear(int(self.conv_dim), int(output_dim))

        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)

    def cumax(self, x, mode="l2r"):
        if mode == "l2r":
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == "r2l":
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x

    def step(self, inputs, c_last, h_last, interval, device):
        x_in = inputs.to(device=device)

        # Integrate inter-visit time intervals
        interval = interval.unsqueeze(-1).to(device=device)
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1)).to(device)
        x_out2 = self.recurrent_kernel(torch.cat((h_last.to(device=device), interval), dim=-1))

        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, : self.levels], "l2r")
        f_master_gate = f_master_gate.unsqueeze(2).to(device=device)
        i_master_gate = self.cumax(x_out[:, self.levels : self.levels * 2], "r2l")
        i_master_gate = i_master_gate.unsqueeze(2)
        x_out = x_out[:, self.levels * 2 :]
        x_out = x_out.reshape(-1, self.levels * 4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, : self.levels]).to(device=device)
        i_gate = torch.sigmoid(x_out[:, self.levels : self.levels * 2]).to(device=device)
        o_gate = torch.sigmoid(x_out[:, self.levels * 2 : self.levels * 3])
        c_in = torch.tanh(x_out[:, self.levels * 3 :]).to(device=device)
        c_last = c_last.reshape(-1, self.levels, self.chunk_size).to(device=device)
        overlap = (f_master_gate * i_master_gate).to(device=device)
        c_out = (
            overlap * (f_gate * c_last + i_gate * c_in)
            + (f_master_gate - overlap) * c_last
            + (i_master_gate - overlap) * c_in
        )
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

    def forward(self, input, device, info=None):
        """extra info is not used here"""
        batch_size, time_step, feature_dim = input.size()
        time = torch.ones(batch_size, time_step)
        c_out = torch.zeros(batch_size, self.hidden_dim)
        h_out = torch.zeros(batch_size, self.hidden_dim)

        tmp_h = (
            torch.zeros_like(h_out, dtype=torch.float32)
            .view(-1)
            .repeat(self.conv_size)
            .view(self.conv_size, batch_size, self.hidden_dim)
        )
        tmp_dis = torch.zeros((self.conv_size, batch_size))
        h = []
        origin_h = []
        distance = []
        for t in range(time_step):
            out, c_out, h_out = self.step(input[:, t, :], c_out, h_out, time[:, t], device)
            cur_distance = 1 - torch.mean(
                out[..., self.hidden_dim : self.hidden_dim + self.levels], -1
            )
            cur_distance_in = torch.mean(out[..., self.hidden_dim + self.levels :], -1)
            origin_h.append(out[..., : self.hidden_dim])
            tmp_h = torch.cat((tmp_h[1:].to(device=device), out[..., : self.hidden_dim].unsqueeze(0).to(device=device)), 0)
            tmp_dis = torch.cat((tmp_dis[1:].to(device=device), cur_distance.unsqueeze(0).to(device=device)), 0)
            distance.append(cur_distance)

            # Re-weighted convolution operation
            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.cumsum(local_dis, dim=1)
            local_dis = torch.softmax(local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)
            local_h = local_h * local_dis.unsqueeze(1)

            # Re-calibrate Progression patterns
            local_theme = torch.mean(local_h, dim=-1)
            local_theme = self.nn_scale(local_theme).to(device)
            local_theme = torch.relu(local_theme)
            local_theme = self.nn_rescale(local_theme).to(device)
            local_theme = torch.sigmoid(local_theme)

            local_h = self.nn_conv(local_h).squeeze(-1)
            local_h = local_theme * local_h
            h.append(local_h)

        origin_h = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)
        if self.dropres > 0.0:
            origin_h = self.nn_dropres(origin_h)
        rnn_outputs = rnn_outputs + origin_h
        rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)

        output = rnn_outputs.contiguous().view(batch_size, time_step, self.hidden_dim)
        # output = self.nn_output(rnn_outputs)
        # output = output.contiguous().view(batch_size, time_step, self.output_dim)
        # output = torch.sigmoid(output)

        # return output, torch.stack(distance)
        return output
