import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RETAIN(nn.Module):
    """
    Ref: https://github.com/Yuyoo/retain/blob/master/retain_torch/Retain_torch.py
    """

    def __init__(
        self,
        dim_input,
        dim_emb,
        dim_alpha,
        dim_beta,
        drop=0.0,
    ):
        super(RETAIN, self).__init__()
        self.dim_input = dim_input
        self.dim_emb = dim_emb
        self.dim_alpha = dim_alpha
        self.dim_beta = dim_beta
        self.drop = drop

        self.embedding = nn.Linear(self.dim_input, self.dim_emb)
        self.dropout = nn.Dropout(self.drop)
        self.gru_alpha = nn.GRU(self.dim_emb, self.dim_alpha)
        self.gru_beta = nn.GRU(self.dim_emb, self.dim_beta)
        self.alpha_att = nn.Linear(self.dim_alpha, 1)
        self.beta_att = nn.Linear(self.dim_beta, self.dim_emb)

    def initHidden_alpha(self, batch_size):
        return torch.zeros(1, batch_size, self.dim_alpha)

    def initHidden_beta(self, batch_size):
        return torch.zeros(1, batch_size, self.dim_beta)

    # 两个attention的处理，其中att_timesteps是目前为止的步数
    # 返回的是一个3维向量，维度为(n_timesteps × n_samples × dim_emb)
    def attentionStep(self, h_a, h_b, att_timesteps):
        """
        两个attention的处理，其中att_timesteps是目前为止的步数
        返回的是一个3维向量，维度为(n_timesteps × n_samples × dim_emb)
        :param h_a:
        :param h_b:
        :param att_timesteps:
        :return:
        """
        reverse_emb_t = self.emb[:att_timesteps].flip(dims=[0])
        reverse_h_a = self.gru_alpha(reverse_emb_t, h_a)[0].flip(dims=[0]) * 0.5
        reverse_h_b = self.gru_beta(reverse_emb_t, h_b)[0].flip(dims=[0]) * 0.5

        preAlpha = self.alpha_att(reverse_h_a)
        preAlpha = torch.squeeze(preAlpha, dim=2)
        alpha = torch.transpose(F.softmax(torch.transpose(preAlpha, 0, 1), dim=1), 0, 1)
        beta = torch.tanh(self.beta_att(reverse_h_b))

        c_t = torch.mean((alpha.unsqueeze(2) * beta * self.emb[:att_timesteps]), dim=0)
        return c_t

    def forward(self, x):
        first_h_a = self.initHidden_alpha(x.shape[1])
        first_h_b = self.initHidden_beta(x.shape[1])

        self.emb = self.embedding(x)
        if self.drop < 1:
            self.emb = self.dropout(self.emb)

        count = np.arange(x.shape[0]) + 1
        self.c_t = torch.zeros_like(self.emb)  # shape=(seq_len, batch_size, day_dim)
        for i, att_timesteps in enumerate(count):
            # 按时间步迭代，计算每个时间步的经attention的gru输出
            self.c_t[i] = self.attentionStep(first_h_a, first_h_b, att_timesteps)

        if self.drop < 1.0:
            self.c_t = self.dropout(self.c_t)

        # # output层
        # y_hat = self.out(self.c_t)
        # y_hat = torch.sigmoid(y_hat)
        return self.c_t
