import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class MWTA_LSTM(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size, seq_len, output_dim, batch_first=True, bidirectional=True):
        super(MWTA_LSTM, self).__init__()
        self.input_size = input_size
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.initializer_range = 0.02
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.c1 = torch.Tensor([1]).float()
        self.c2 = torch.Tensor([np.e]).float()
        self.ones = torch.ones([self.input_size, 1, self.hidden_size]).float()
        self.decay_features = torch.Tensor(torch.arange(self.input_size)).float()
        self.register_buffer('c1_const', self.c1)
        self.register_buffer('c2_const', self.c2)
        self.register_buffer("ones_const", self.ones)

        self.U_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.Dw = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))

        self.W_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_decomp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))

        self.W_cell_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        self.b_decomp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        # Output layer
        self.F_alpha = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size * 2, 1)))
        self.F_alpha_n_b = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1)))
        self.F_beta = nn.Linear(self.seq_len, 4 * self.hidden_size)
        self.layer_norm1 = nn.LayerNorm([self.input_size, self.seq_len])
        self.layer_norm = nn.LayerNorm([self.input_size, 4 * self.hidden_size])
        self.Phi = nn.Linear(4 * self.hidden_size, self.output_dim)

    @torch.jit.script_method
    def TLSTM_unit(self, prev_hidden_memory, cell_hidden_memory, inputs, times):
        h_tilda_t, c_tilda_t = prev_hidden_memory, cell_hidden_memory,
        x = inputs
        t = times
        T = self.map_elapse_time(t)
        C_ST = torch.tanh(torch.einsum("bij,ijk->bik", c_tilda_t, self.W_decomp))
        C_ST_dis = torch.mul(T, C_ST)
        c_tilda_t = c_tilda_t - C_ST + C_ST_dis
        # Time Gate 
        t_gate = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_time) +
                               torch.sigmoid(self.map_elapse_time(t)) + self.b_time)
        # Input Gate
        i = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_i) + \
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                          c_tilda_t * self.W_cell_i + self.b_i)
        # Forget Gate
        f = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_f) + \
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                          c_tilda_t * self.W_cell_f + self.b_f)
        # Candidate Memory Cell
        C = torch.tanh(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_c) + \
                       torch.einsum("bij,ijk->bik", h_tilda_t, self.W_c) + self.b_c)
        # Current Memory Cell---> Modelling Intervention's effects 
        Ct = (f + t_gate) * c_tilda_t + i * t_gate * C
        # Output Gate
        o = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_o) +
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                          t_gate + Ct * self.W_cell_o + self.b_o)
        # Current Hidden State
        h_tilda_t = o * torch.tanh(Ct)

        return h_tilda_t, Ct

    @torch.jit.script_method
    def map_elapse_time(self, t):
        T = torch.div(self.c1_const, torch.log(t + self.c2_const))
        T = torch.einsum("bij,jik->bjk", T.unsqueeze(1), self.ones_const)
        return T

    @torch.jit.script_method
    def forward(self, inputs, times):
        device = inputs.device
        if self.batch_first:
            batch_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            batch_size = inputs.size()[1]
        prev_hidden = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
        prev_cell = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)

        seq_len = inputs.size()[0]
        hidden_his = torch.jit.annotate(List[Tensor], [])
        for i in range(seq_len):
            prev_hidden, prev_cell = self.TLSTM_unit(prev_hidden, prev_cell, inputs[i], times[i])
            hidden_his += [prev_hidden]
        hidden_his = torch.stack(hidden_his)
        if self.bidirectional:
            second_hidden = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
            second_cell = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
            second_inputs = torch.flip(inputs, [0])
            second_times = torch.flip(times, [0])
            second_hidden_his = torch.jit.annotate(List[Tensor], [])
            for i in range(seq_len):
                if i == 0:
                    time = times[i]
                else:
                    time = second_times[i - 1]
                second_hidden, second_cell = self.TLSTM_unit(second_hidden, second_cell,
                                                             second_inputs[i], time)
                second_hidden_his += [second_hidden]
            second_hidden_his = torch.stack(second_hidden_his)
            hidden_his = torch.cat((hidden_his, second_hidden_his), dim=-1)
            prev_hidden = torch.cat((prev_hidden, second_hidden), dim=-1)
            prev_cell = torch.cat((prev_cell, second_cell), dim=-1)
        if self.batch_first:
            hidden_his = hidden_his.permute(1, 0, 2, 3)
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", hidden_his, self.F_alpha) + self.F_alpha_n_b)
        alphas = alphas.reshape(alphas.size(0), alphas.size(2), alphas.size(1) * alphas.size(-1))
        mu = self.Phi(self.layer_norm(self.F_beta(self.layer_norm1(alphas))))
        out = torch.max(mu, dim=1).values
        return out, (prev_hidden, prev_cell)


class TimeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, dropout=0.2):
        super(TimeLSTM, self).__init__()
        # hidden dimensions
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        # Temporal embedding MWTA_LSTM
        self.mwta_lstm = MWTA_LSTM(self.input_size, self.hidden_size, self.seq_len, self.output_dim)

    def forward(self, historic_features, timestamp, last_features, features_freqs, is_test=False):
        # Temporal features embedding
        # outputs, prev_hidden = self.mwta_lstm(historic_features, timestamp, last_features, features_freqs)
        outputs, prev_hidden = self.mwta_lstm(historic_features, timestamp)
        if is_test:
            return prev_hidden, outputs
        else:
            return outputs
