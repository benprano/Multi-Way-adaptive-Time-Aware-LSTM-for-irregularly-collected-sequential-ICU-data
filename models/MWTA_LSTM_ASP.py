import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import List
from torch import Tensor
from copy import deepcopy
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def seed_all(seed: int = 1992):
    """Seed all random number generators."""
    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
seed_all()

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        super().__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.scale = torch.sqrt(torch.FloatTensor([input_size]))
        self.seq_length = seq_len

    def forward(self, x):
        # Calculate Query, Key, and Value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute scaled dot-product attention weights
        score = torch.matmul(Q, K.transpose(1, 2))
        d_k = x.shape[-1]
        scaling_factor = torch.sqrt(torch.tensor(d_k).to(torch.float32))
        score = score / scaling_factor
        attn_weights = nn.functional.softmax(score, dim=-1)

        # Apply attention weights to Value
        output = torch.matmul(attn_weights, V)

        return output

class Attention(nn.Module):
    def __init__(self, input_size, seq_len):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.attention = SelfAttention(self.input_size, self.seq_len)
        self.out_layer = nn.Linear(self.input_size, self.input_size)

    def forward(self, x):
        attn_output = self.attention(x)
        residual = x + attn_output
        output = self.out_layer(residual)
        return output, attn_output


class GLU(nn.Module):
    """
      The Gated Linear Unit GLU(a,b) = mult(a,sigmoid(b)) is common in NLP
      architectures like the Gated CNN. Here sigmoid(b) corresponds to a gate
      that controls what information from a is passed to the following layer.

      Args:
          input_size (int): number defining input and output size of the gate
    """

    def __init__(self, input_size):
        super().__init__()
        self.initializer_range = 0.02
        # Input
        self.a = nn.Linear(input_size, input_size)

        # Gate
        self.sigmoid = nn.Sigmoid()
        self.b = nn.Linear(input_size, input_size)

    def forward(self, x):
        """
        Args:
            x (torch.tensor): tensor passing through the gate
        """
        gate = self.sigmoid(self.b(x))
        x = self.a(x)

        return torch.mul(gate, x)


class TemporalLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.

        Similar to TimeDistributed in Keras, it is a wrapper that makes it possible
        to apply a layer to every temporal slice of an input.
        """
        self.module = module

    def forward(self, x):
        """
        Args:
            x (torch.tensor): tensor with time steps to pass through the same layer.
        """
        t, n = x.size(0), x.size(1)
        x = x.reshape(t * n, -1)
        x = self.module(x)
        x = x.reshape(t, n, x.size(-1))

        return x


class ScaledDotProductAttention(nn.Module):
    """
    Attention mechansims usually scale values based on relationships between
    keys and queries.

    Attention(Q,K,V) = A(Q,K)*V where A() is a normalization function.

    A common choice for the normalization function is scaled dot-product attention:

    A(Q,K) = Softmax(Q*K^T / sqrt(d_attention))

    Args:
          dropout (float): Fraction between 0 and 1 corresponding to the degree of dropout used
    """

    def __init__(self, dropout=0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, mask=None):
        """
        Args:
          query (torch.tensor):
          key (torch.tensor):
          value (torch.tensor):
          mask (torch.tensor):
        """

        d_k = key.shape[-1]
        scaling_factor = torch.sqrt(torch.tensor(d_k).to(torch.float32))

        scaled_dot_product = torch.matmul(query, key.permute(0, 2, 1)) / scaling_factor
        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -1e9)
        attention = self.softmax(scaled_dot_product)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)

        return output, attention


class InterpretableMultiHeadAttention(nn.Module):
    """
    Different attention heads can be used to improve the learning capacity of
    the model.

    MultiHead(Q,K,V) = [H_1, ..., H_m]*W_H
    H_h = Attention(Q*Wh_Q, K*Wh_K, V*Wh_V)

    Each head has specific weights for keys, queries and values. W_H linearly
    combines the concatenated outputs from all heads.

    To increase interpretability, multi-head attention has been modified to share
    values in each head.

    InterpretableMultiHead(Q,K,V) = H_I*W_H
    H_I = 1/H * SUM(Attention(Q*Wh_Q, K*Wh_K, V*W_V)) # Note that W_V does not depend on the head.

    Args:
          num_heads (int): Number of attention heads
          hidden_size (int): Hidden size of the model
          dropout (float): Fraction between 0 and 1 corresponding to the degree of dropout used
    """

    def __init__(self, num_attention_heads, hidden_size, dropout=0.0):
        super().__init__()
        self.initializer_range = 0.02
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.qs = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for i in range(self.num_attention_heads)])
        self.ks = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for i in range(self.num_attention_heads)])

        vs_layer = nn.Linear(self.hidden_size, self.hidden_size,
                             bias=False)  # Value is shared for improved interpretability
        self.vs = nn.ModuleList([vs_layer for i in range(self.num_attention_heads)])

        self.attention = ScaledDotProductAttention()
        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, query, key, value, mask=None):

        b_size, tgt_len, embed_dim = query.shape
        head_dim = embed_dim // self.num_attention_heads

        # Now we iterate over each head to calculate outputs and attention
        heads = []
        attentions = []

        for i in range(self.num_attention_heads):
            q_i = self.qs[i](query)
            k_i = self.ks[i](key)
            v_i = self.vs[i](value)

            # Reshape q, k, v for multihead attention
            q_i = query.reshape(b_size, tgt_len, self.num_attention_heads, head_dim).transpose(1, 2).reshape(
                batch_size * self.num_attention_heads, tgt_len, head_dim)
            k_i = key.reshape(b_size, tgt_len, self.num_attention_heads, head_dim).transpose(1, 2).reshape(
                batch_size * self.num_attention_heads, tgt_len, head_dim)
            v_i = value.reshape(b_size, tgt_len, self.num_attention_heads, head_dim).transpose(1, 2).reshape(
                batch_size * self.num_attention_heads, tgt_len, head_dim)

            head, attention = self.attention(q_i, k_i, v_i, mask)

            # Revert to original target shape
            head = head.reshape(batch_size, self.num_attention_heads,
                                tgt_len, head_dim).transpose(1, 2).reshape(-1,tgt_len,self.num_attention_heads * head_dim)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attentions.append(attention)

        # Output the results
        if self.num_attention_heads > 1:
            heads = torch.stack(heads, dim=2)  # .reshape(batch_size, tgt_len, -1, self.hidden_size)
            outputs = torch.mean(heads, dim=2)
        else:
            outputs = head

        attentions = torch.stack(attentions, dim=2)
        attention = torch.mean(attentions, dim=2)

        outputs = self.linear(outputs)
        outputs = self.dropout(outputs)

        return outputs + query, attention

#    ASP POOLING

class ASP(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.pool_size = kernel_size

    def forward(self, x):
        # x.shape: (batch_size, channels, sequence_length)
        b_size, channels, sequence_length = x.shape

        # Flatten the input tensor
        x_flat = x.view(b_size, channels, -1)

        # Compute the number of pools and pool size
        num_pools = sequence_length // self.pool_size
        pool_size = sequence_length // num_pools

        # Reshape x_flat into (batch_size, channels, num_pools, pool_size)
        x_reshaped = x_flat.view(b_size, channels, num_pools, pool_size)

        # Compute the mean and meadian absolute deviation of each pool
        mean = torch.mean(x_reshaped, dim=3, keepdim=True)
        # Calculate MAD
        median = torch.median(x_reshaped, dim=3, keepdim=True).values
        mad = torch.median(torch.abs(x_reshaped - median), dim=3, keepdim=True).values
        # Sample from a normal distribution with the computed mean and mad
        pooled = mean + mad * torch.randn_like(mean)

        # Reshape pooled back into (batch_size, channels, sequence_length)
        pooled_reshaped = pooled.view(b_size, channels, -1)

        return pooled_reshaped


# MWTA LSTM ASP

class MWTALSTM(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size, batch_first=True, bidirectional=True):
        super(MWTALSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initializer_range = 0.02
        self.bidirectional = bidirectional
        self.c1 = torch.Tensor([1]).float()
        self.c2 = torch.Tensor([np.e]).float()
        self.ones = torch.ones([self.input_size, 1, self.hidden_size]).float()
        self.decay_features = torch.Tensor(torch.arange(self.input_size)).float()
        self.register_buffer('c1_const', self.c1)
        self.register_buffer('c2_const', self.c2)
        self.register_buffer("ones_const", self.ones)
        self.alpha = torch.FloatTensor([0.5])
        self.alpha_imp = torch.FloatTensor([0.5])
        self.register_buffer("factor", self.alpha)
        self.register_buffer("features_decay", self.decay_features)
        self.register_buffer("factor_impu", self.alpha_imp)

        self.U_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
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
        self.b_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        # Gate Linear Unit for last records
        self.activation_layer = nn.ELU()

    @torch.jit.script_method
    def tlstm_unit(self, prev_hidden_memory, cell_hidden_memory, inputs, times, last_data, freq_list):
        h_tilda_t, c_tilda_t = prev_hidden_memory, cell_hidden_memory,
        x = inputs
        t = times
        l = last_data
        freq = freq_list
        T = self.map_elapse_time(t)

        last_tilda_t = self.activation_layer(torch.einsum("bij,jik->bjk", l.unsqueeze(1),
                                                          self.U_last) + self.b_last)
        C_ST = torch.tanh(torch.einsum("bij,ijk->bik", c_tilda_t, self.W_decomp))
        C_ST_dis = torch.mul(T, C_ST)
        c_tilda_t = c_tilda_t - C_ST + C_ST_dis
        h_tilda_t = h_tilda_t + last_tilda_t
        # Ajust previous to incoporate the latest records for each feature
        h_tilda_t = h_tilda_t + last_tilda_t
        # Time Gate
        t_gate = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_time) +
                               torch.sigmoid(self.map_elapse_time(t)) + self.b_time)
        # Input Gate
        i = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_i) + \
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                          c_tilda_t * self.W_cell_i + self.b_i * self.freq_decay(freq))
        # Forget Gate
        f = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_f) + \
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                          c_tilda_t * self.W_cell_f + self.b_f)

        f_new = f * self.map_elapse_time(t) + (1 - f) * self.freq_decay(freq)
        # Candidate Memory Cell
        C = torch.tanh(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_c) + \
                       torch.einsum("bij,ijk->bik", h_tilda_t, self.W_c) + self.b_c)
        # Current Memory Cell
        Ct = (f_new + t_gate) * c_tilda_t + i * t_gate * C
        # Output Gate
        o = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_o) +
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                          t_gate + last_tilda_t + Ct * self.W_cell_o + self.b_o)
        # Current Hidden State
        h_tilda_t = o * torch.tanh(Ct + last_tilda_t)

        return h_tilda_t, Ct, self.freq_decay(freq), f_new

    @torch.jit.script_method
    def map_elapse_time(self, t):
        T = torch.div(self.c1_const, torch.log(t + self.c2_const))
        T = torch.einsum("bij,jik->bjk", T.unsqueeze(1), self.ones_const)
        return T

    @torch.jit.script_method
    def freq_decay(self, freq_dict: torch.Tensor):
        freq_weight = torch.exp(-self.factor_impu * freq_dict)
        weights = torch.sigmoid(torch.einsum("bij,jik->bjk", freq_weight.unsqueeze(-1), self.Dw) + self.b_d)
        return weights

    @torch.jit.script_method
    def forward(self, inputs, times, last_values, freqs):
        device_ = inputs.device
        if self.batch_first:
            b_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            last_values = last_values.permute(1, 0, 2)
            freqs = freqs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            b_size = inputs.size()[1]
        prev_hidden = torch.zeros((b_size, inputs.size()[2], self.hidden_size), device=device_)
        prev_cell = torch.zeros((b_size, inputs.size()[2], self.hidden_size), device=device_)

        seq_len = inputs.size()[0]
        hidden_his = torch.jit.annotate(List[Tensor], [])
        weights_decay = torch.jit.annotate(List[Tensor], [])
        weights_fgate = torch.jit.annotate(List[Tensor], [])
        for i in range(seq_len):
            prev_hidden, prev_cell, pre_we_decay, fgate_f = self.tlstm_unit(prev_hidden, prev_cell,
                                                                            inputs[i], times[i],
                                                                            last_values[i], freqs[i])
            hidden_his += [prev_hidden]
            weights_decay += [pre_we_decay]
            weights_fgate += [fgate_f]
        hidden_his = torch.stack(hidden_his)
        weights_decay = torch.stack(weights_decay)
        weights_fgate = torch.stack(weights_fgate)
        if self.bidirectional:
            second_hidden = torch.zeros((b_size, inputs.size()[2], self.hidden_size), device=device)
            second_cell = torch.zeros((b_size, inputs.size()[2], self.hidden_size), device=device)
            second_inputs = torch.flip(inputs, [0])
            second_times = torch.flip(times, [0])
            second_hidden_his = torch.jit.annotate(List[Tensor], [])
            second_weights_decay = torch.jit.annotate(List[Tensor], [])
            second_weights_fgate = torch.jit.annotate(List[Tensor], [])
            for i in range(seq_len):
                if i == 0:
                    time = times[i]
                else:
                    time = second_times[i - 1]
                second_hidden, second_cell, b_we_decay, fgate_b = self.tlstm_unit(second_hidden, second_cell,
                                                                                  second_inputs[i], time,
                                                                                  last_values[i], freqs[i])
                second_hidden_his += [second_hidden]
                second_weights_decay += [b_we_decay]
                second_weights_fgate += [fgate_b]
            second_hidden_his = torch.stack(second_hidden_his)
            second_weights_fgate = torch.stack(second_weights_fgate)
            second_weights_decay = torch.stack(second_weights_decay)
            weights_decay = torch.cat((weights_decay, second_weights_decay), dim=-1)
            weights_fgate = torch.cat((weights_fgate, second_weights_fgate), dim=-1)
            hidden_his = torch.cat((hidden_his, second_hidden_his), dim=-1)
        if self.batch_first:
            hidden_his = hidden_his.permute(1, 0, 2, 3)
            weights_decay = weights_decay.permute(1, 0, 2, 3)
            weights_fgate = weights_fgate.permute(1, 0, 2, 3)
        return torch.mean(hidden_his, dim=2), (weights_decay, weights_fgate)


class MwtaLstmAsp(nn.Module):
    def __init__(self, seq_len, statics_size, input_size, hidden_size, output_size, dropout=0.2):
        super(MwtaLstmAsp, self).__init__()
        # hidden dimensions
        self.initializer_range = 0.02

        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.output_size = output_size
        self.statics_dim = statics_size
        self.length_seq = seq_len + 1
        self.seq_length = seq_len
        self.dropout_rate = dropout
        self.mwtalstm = MWTALSTM(self.input_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.statics_emb = nn.Sequential(nn.Linear(self.statics_dim, self.hidden_dim),
                                         nn.ReLU(),
                                         nn.Dropout(0.1),
                                         nn.Linear(self.hidden_dim, self.hidden_dim * 2))
        self.statics_bn = nn.BatchNorm1d(num_features=self.hidden_dim * 2)
        # Temporal Self-attention layer
        self.multihead_attn = InterpretableMultiHeadAttention(4, self.hidden_dim * 2)
        self.attention_gated_skip_connection = TemporalLayer(GLU(self.hidden_dim * 2))
        self.attention = Attention(self.hidden_dim * 2, self.seq_length)
        self.elu = nn.ELU()
        self.attention_add_norm = TemporalLayer(nn.LayerNorm(self.hidden_dim * 2,
                                                             eps=1e-12))
        # Conv1D Embedding
        self.window_sizes = [w for w in range(1, seq_len + 1) if w % 6 == 0]
        self.cnn_out = self.hidden_dim
        self.max_text_len = seq_len + 1
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.hidden_dim * 2,
                                    out_channels=self.cnn_out,
                                    kernel_size=h), nn.ELU(),
                          ASP(kernel_size=self.max_text_len - h + 1))
            for h in self.window_sizes
        ])
        # Conv1D & Output Layers
        self.conv1d_bn = nn.BatchNorm1d(num_features=self.cnn_out * len(self.window_sizes))
        self.fc = nn.Linear(in_features=self.cnn_out * len(self.window_sizes),
                            out_features=self.output_size)

    def forward(self, statics_features, historic_features, timestamp, last_features,
                features_freqs, is_test=False):
        # Temporal features embedding
        outputs, hidden = self.mwtalstm(historic_features, timestamp,
                                        last_features, features_freqs)
        outputs = self.elu(outputs)
        multihead_outputs, multihead_attention = self.multihead_attn(outputs, outputs, outputs)
        outputs_blocks, attentions = self.attention(multihead_outputs)
        attention_gated_outputs = self.attention_gated_skip_connection(outputs_blocks)
        attention_outputs = self.attention_add_norm(attention_gated_outputs)
        attention_outputs = F.dropout(attention_outputs, p=self.dropout_rate)
        # Statics Embedding
        statics_features = self.statics_emb(statics_features)
        statics_features = self.statics_bn(statics_features)
        # Combined Temporal & Statics outputs
        combined_features = torch.cat((attention_outputs.permute(0, 2, 1),
                                       statics_features.unsqueeze(-1)),
                                      dim=2)
        out = [conv(combined_features) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.conv1d_bn(out)
        out = self.fc(out)
        if is_test:
            return multihead_attention, out
        else:
            return out


class MwtaLSTM(nn.Module):
    def __init__(self, seq_len, statics_size, input_size, hidden_size, output_size, dropout=0.2):
        super(MwtaLSTM, self).__init__()
        # hidden dimensions
        self.initializer_range = 0.02
        self.statics_dim = statics_size
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.output_size = output_size
        self.length_seq = seq_len + 1
        self.seq_length = seq_len
        self.dropout_rate = dropout
        self.mwtalstm = MWTALSTM(self.input_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.statics_emb = nn.Sequential(nn.Linear(self.statics_dim, self.hidden_dim),
                                         nn.ReLU(),
                                         nn.Dropout(0.1),
                                         nn.Linear(self.hidden_dim, self.hidden_dim * 2))
        self.statics_bn = nn.BatchNorm1d(num_features=self.hidden_dim * 2)
        # Temporal Self-attention layer
        self.multihead_attn = InterpretableMultiHeadAttention(4, self.hidden_dim * 2)
        self.attention_gated_skip_connection = TemporalLayer(GLU(self.hidden_dim * 2))
        self.elu = nn.ELU()
        self.attention_add_norm = TemporalLayer(nn.LayerNorm(self.hidden_dim * 2,
                                                             eps=1e-12))
        # Conv1D Embedding
        self.window_sizes = [w for w in range(1, seq_len + 1) if w % 6 == 0]
        self.cnn_out = self.hidden_dim
        self.max_text_len = seq_len + 1
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.hidden_dim * 2,
                                    out_channels=self.cnn_out,
                                    kernel_size=h), nn.ELU(),
                          ASP(kernel_size=self.max_text_len - h + 1))
            for h in self.window_sizes
        ])
        # Conv1D & Output Layers
        self.conv1d_bn = nn.BatchNorm1d(num_features=self.cnn_out * len(self.window_sizes))
        self.fc = nn.Linear(in_features=self.cnn_out * len(self.window_sizes),
                            out_features=self.output_size)

    def forward(self, statics_features, historic_features, timestamp, last_features,
                features_freqs, is_test=False):
        # Temporal features embedding
        outputs, hidden = self.mwtalstm(historic_features, timestamp,
                                        last_features, features_freqs)
        outputs = self.elu(outputs)
        multihead_outputs, multihead_attention = self.multihead_attn(outputs, outputs, outputs)
        attention_gated_outputs = self.attention_gated_skip_connection(multihead_outputs)
        attention_outputs = self.attention_add_norm(attention_gated_outputs)
        attention_outputs = F.dropout(attention_outputs, p=self.dropout_rate)
        # Statics Embedding
        statics_features = self.statics_emb(statics_features)
        statics_features = self.statics_bn(statics_features)
        # Combined Temporal & Statics outputs
        combined_features = torch.cat((attention_outputs.permute(0, 2, 1),
                                       statics_features.unsqueeze(-1)),
                                      dim=2)
        out = [conv(combined_features) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.conv1d_bn(out)
        out = self.fc(out)
        if is_test:
            return multihead_attention, out
        else:
            return out


class EarlyStopping:
    def __init__(self, mode, path, patience=3, delta=0):
        if mode not in {'min', 'max'}:
            raise ValueError("Argument mode must be one of 'min' or 'max'.")
        if patience <= 0:
            raise ValueError("Argument patience must be a positive integer.")
        if delta < 0:
            raise ValueError("Argument delta must not be a negative number.")

        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.counter = 0

    def _is_improvement(self, val_score):
        """Return True iff val_score is better than self.best_score."""
        if self.mode == 'max' and val_score > self.best_score + self.delta:
            return True
        elif self.mode == 'min' and val_score < self.best_score - self.delta:
            return True
        return False

    def __call__(self, val_score, model):
        """
        Return True iff self.counter >= self.patience.
        """

        if self._is_improvement(val_score):
            self.best_score = val_score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print("Val loss improved, Saving model's best weights.")
            return False
        else:
            self.counter += 1
            print(f'Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                print(f'Stopped early. Best val loss: {self.best_score:.4f}')
                return True
            return None


class TrainerHelpers:
    def __init__(self, input_size, hidden_size, seq_len, output_size, device_, optim_, loss_criterion, schedulers,
                 num_epochs, patience_n=50, task=True):
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.seq_length = seq_len
        self.output_dim = output_size
        self.device = device_
        self.optim = optim_
        self.loss_criterion = loss_criterion
        self.schedulers = schedulers
        self.num_epochs = num_epochs
        self.patience_n = patience_n
        self.task = task

    @staticmethod
    def acc(predicted, label):
        predicted = predicted.sigmoid()
        pred = torch.round(predicted.squeeze())
        return torch.sum(pred == label.squeeze()).item()

    def train_model(self, model, train_dataloader):
        model.train()
        running_loss, running_corrects, mae_train = 0.0, 0, 0
        for bi, inputs in enumerate(tqdm(train_dataloader, total=len(train_dataloader), leave=False)):
            s_features, temporal_features, timestamp, last_obser_data, data_freqs, labels = inputs

            s_features = s_features.to(torch.float32).to(device)
            temporal_features = temporal_features.to(torch.float32).to(device)
            timestamp = timestamp.to(torch.float32).to(device)
            last_obser_data = last_obser_data.to(torch.float32).to(device)
            data_freqs = data_freqs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            self.optim.zero_grad()
            outputs = model(s_features, temporal_features, timestamp,
                            last_obser_data, data_freqs)

            loss = self.loss_criterion(outputs.sigmoid(), labels)
            loss.backward()
            self.optim.step()
            running_loss += loss.item()
            running_corrects += self.acc(outputs, labels)
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_corrects / len(train_dataloader.dataset)
        return epoch_loss, epoch_acc

    def valid_model(self, model, valid_dataloader):
        model.eval()
        running_loss, running_corrects, mae_val = 0.0, 0, 0
        fin_targets, fin_outputs = [], []
        for bi, inputs in enumerate(tqdm(valid_dataloader, total=len(valid_dataloader), leave=False)):
            s_features, temporal_features, timestamp, last_obser_data, data_freqs, labels = inputs

            s_features = s_features.to(torch.float32).to(device)
            temporal_features = temporal_features.to(torch.float32).to(device)
            timestamp = timestamp.to(torch.float32).to(device)
            last_obser_data = last_obser_data.to(torch.float32).to(device)
            data_freqs = data_freqs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            with torch.no_grad():
                outputs = model(s_features, temporal_features, timestamp,
                                last_obser_data, data_freqs)
            loss = self.loss_criterion(outputs.sigmoid(), labels)
            running_loss += loss.item()
            running_corrects += self.acc(outputs, labels)
            fin_targets.append(labels.cpu().detach().numpy())
            fin_outputs.append(outputs.cpu().detach().numpy())
        epoch_loss = running_loss / len(valid_dataloader)
        epoch_accuracy = running_corrects / len(valid_dataloader.dataset)
        return epoch_loss, epoch_accuracy, np.vstack(fin_targets), np.vstack(fin_outputs)

    def eval_model(self, model_class, model_path, test_dataloader):
        # Initialize the model architecture
        model = model_class(seq_length, statics_dim, input_dim, hidden_dim, output_dim).to(self.device)
        # Load the model weights
        model.load_state_dict(torch.load(model_path, weights_only=True))
        # Set the model to evaluation mode
        model.eval()
        fin_targets, fin_outputs = [], []
        for bi, inputs in enumerate(tqdm(test_dataloader, total=len(test_dataloader), leave=False,
                                         desc='Evaluating on test data')):
            s_features, temporal_features, timestamp, last_obser_data, data_freqs, labels = inputs

            s_features = s_features.to(torch.float32).to(device)
            temporal_features = temporal_features.to(torch.float32).to(device)
            timestamp = timestamp.to(torch.float32).to(device)
            last_obser_data = last_obser_data.to(torch.float32).to(device)
            data_freqs = data_freqs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            with torch.no_grad():
                outputs = model(s_features, temporal_features, timestamp,
                                last_obser_data, data_freqs)

            fin_outputs.append(outputs.sigmoid().cpu().detach().numpy())
            fin_targets.append(labels.cpu().detach().numpy())
        return np.vstack(fin_targets), np.vstack(fin_outputs)

    def train_validate_evaluate(self, model_class, model, modelname, train_dataloader, val_loader, test_data_loader, params,
                                model_path):
        best_losses, all_scores = [], []
        es = EarlyStopping(mode='min', path=f"{os.path.join(model_path, f'model_{modelname}.pth')}",
                           patience=self.patience_n)
        for epoch in range(self.num_epochs):
            if self.task:
                loss, accuracy = self.train_model(model, train_dataloader)
                eval_loss, eval_accuracy, __, _ = self.valid_model(model, val_loader)
                if self.schedulers is not None:
                    self.schedulers.step(epoch)
                print(
                    f"lr: {self.optim.param_groups[0]['lr']:.7f}, epoch: {epoch + 1}/{self.num_epochs}, train loss: {loss:.8f}, acc: {accuracy:.8f} | valid loss: {eval_loss:.8f}, acc: {eval_accuracy:.4f}")
                if es(eval_loss, model):
                    best_losses.append(es.best_score)
                    print("best_score", es.best_score)
                    break
            else:
                loss = self.train_model(model, train_dataloader)
                eval_loss, mse_loss, mae_loss, _, _ = self.valid_model(model, val_loader)
                if self.schedulers is not None:
                    self.schedulers.step(epoch)
                print(
                    f"lr: {self.optim.param_groups[0]['lr']:.7f}, epoch: {epoch + 1}/{self.num_epochs}, train loss: {loss:.8f} |  valid loss: {eval_loss:.8f} valid mse loss: {mse_loss:.8f}, valid mae loss: {mae_loss:.8f}")
                if es(mse_loss, model):
                    best_losses.append(es.best_score)
                    print("best_score", es.best_score)
                    break
        if self.task:
            _, _, y_true, y_pred = self.valid_model(model, val_loader)
            print(y_true.shape, y_pred.shape)
            pr_score = average_precision_score(y_true, y_pred)
            print(f"[INFO] PR-AUC ON FOLD :{modelname} -  score val data: {pr_score:.4f}")
        else:
            y_true, y_pred = self.valid_model(model, val_loader)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            print(
                f"[INFO] mse loss & mae loss on validation data Fold {modelname}: mse loss: {mse:.8f} - mae loss: {mae:.8f}")
        if self.task:
            f1_scores_folds = []
            targets, outputs, = self._evaluate_model(model_class,
                                                     f"{os.path.join(model_path, f'model_{modelname}.pth')}",
                                                     test_data_loader)

            delta, f1_scr = self.best_threshold(np.vstack(targets), np.vstack(outputs))
            score = self.metrics_binary(targets, outputs)

            f1_scores_folds.append((delta, f1_scr))
            all_scores.append([score, f1_scores_folds])

            np.savez(os.path.join(model_path, f"results_data_{modelname}.npz"),
                     auc_pr=score, true_labels_data=np.vstack(outputs),
                     predicted_labels_data=np.vstack(targets),
                     folds_f1_scores=f1_scores_folds)
            print(f"[INFO] Results on test Folds {all_scores}")
        else:
            targets, outputs = self._evaluate_model(model_class,
                                                    f"{os.path.join(model_path, f'model_{modelname}.pth')}",
                                                    test_data_loader)
            score = self.metrics_reg(targets, outputs, params)
            all_scores.append([score])
            np.savez(os.path.join(model_path, f"test_data_fold_{modelname}.npz"),
                     reg_scores=score, true_labels=targets, predicted_labels=outputs)
            print(f"[INFO] Results on test Folds {all_scores}")
        return all_scores

    def _evaluate_model(self, model_class, model_path, test_dataloader):
        targets, predicted, all_decays, fgate_weights = [], [], [], []
        y_pred, y_true = self.eval_model(model_class, model_path, test_dataloader)
        targets.append(y_true)
        predicted.append(y_pred)

        targets_all = [np.vstack(targets[i]) for i in range(len(targets))]
        predicted_all = [np.vstack(predicted[i]) for i in range(len(predicted))]
        return targets_all, predicted_all

    @staticmethod
    def metrics_binary(targets, predicted):
        score = []
        for y_true, y_pred, in zip(targets, predicted):
            fpr, tpr, thresholds = roc_curve(y_pred, y_true)
            auc_score = auc(fpr, tpr)
            pr_score = average_precision_score(y_pred, y_true)
            score.append([np.round(np.mean(auc_score), 4),
                           np.round(np.mean(pr_score), 4)])
        return score

    @staticmethod
    def best_threshold(y_train, train_preds):
        delta, tmp = 0, [0, 0, 0]  # idx, cur, max
        for tmp[0] in tqdm(np.arange(0.1, 1.01, 0.01)):
            tmp[1] = f1_score(train_preds, np.array(y_train) > tmp[0])
            if tmp[1] > tmp[2]:
                delta = tmp[0]
                tmp[2] = tmp[1]
        print('best threshold is {:.2f} with F1 score: {:.4f}'.format(delta, tmp[2]))
        return delta, tmp[2]

    @staticmethod
    def adjusted_r2(actual: np.ndarray, predicted: np.ndarray, rowcount: np.int64, featurecount: np.int64):
        return 1 - (1 - r2_score(actual, predicted)) * (rowcount - 1) / (rowcount - featurecount)

    def metrics_reg(self, targets, predicted, rescale_params):
        score = []
        for y_true, y_pred, in zip(targets, predicted):
            target_max, target_min = rescale_params['data_targets_max'], rescale_params['data_targets_min']
            targets_y_true = y_true * (target_max - target_min) + target_min
            targets_y_pred = y_pred * (target_max - target_min) + target_min
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            n = y_true.shape[0]
            r2 = r2_score(targets_y_true, targets_y_pred)
            adj_r2 = self.adjusted_r2(targets_y_true, targets_y_pred, n, self.input_dim)
            score.append([rmse, mae, r2, adj_r2])
        return score

dn="--------"
task_dataset ="-----" # Dataset path

all_dataset_loader = np.load(os.path.join(os.path.join(dn,task_dataset), "train_test_data.npz"),
                                          allow_pickle=True)
dataset_settings = np.load(os.path.join(os.path.join(dn,task_dataset),"data_max_min.npz"),
                                         allow_pickle=True)
train_val_loader = all_dataset_loader['folds_data_train_valid']
test_loader = all_dataset_loader['folds_data_test']
data_max, data_min= dataset_settings['data_max'], dataset_settings['data_min']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim, output_dim  = 128, dataset_settings['output_size'].item()
seq_length = dataset_settings['seq_length'].item()
statics_dim = dataset_settings['stat_dim'].item()
input_dim = dataset_settings['input_dim'].item()

LEARNING_RATE = 1e-3
optimizer_config = {
    "lr": 3e-4,                     # Lower learning rate for stability
    "betas": (0.9, 0.95),           # Slightly less aggressive momentum
    "eps": 1e-8,                    # Increased epsilon avoids NaN
    "weight_decay": 1e-5            # Lower weight decay to prevent underfitting
}
NUM_EPOCHS =500
NUM_FOLDS = 10
model_name="AMITA2i".lower()
n_patience = 100
batch_size=64

mwta = MwtaLSTM(seq_length, statics_dim, input_dim,
                hidden_dim, output_dim).to(device)
# Create an instance of the class
optimizer = torch.optim.Adam(mwta.parameters(), **optimizer_config)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=100, # Number of epochs before restart
    T_mult=2, # Increases T_0 after each restart
    eta_min=1e-6
)
mwta.apply(initialize_weights)
criterion = nn.BCELoss().to(device)
best_model_wts = deepcopy(mwta.state_dict())
print("mwta asp", mwta)

hours_data_used = seq_length
taskname=f"ICU_MORT_ALL_FEATURES_{hours_data_used}_HRS_DATA"
main_path = "RESULTS/"
task_path=f"{os.path.join(main_path, f'{taskname}')}"
if not os.path.exists(task_path):
    os.makedirs(task_path)

train_valid_inference = TrainerHelpers(input_dim, hidden_dim, seq_length, output_dim,
                                       device, optimizer, criterion, scheduler, NUM_EPOCHS,
                                       patience_n=n_patience, task=True)
scores_folds= []
for idx, (train_loader, test_data) in enumerate(zip(train_val_loader ,test_loader)):
    print(f'[INFO]: Training on fold : {idx+1}')
    # Reset the model weights
    mwta.load_state_dict(best_model_wts)
    train_data, valid_data= train_loader
    scores= train_valid_inference.train_validate_evaluate(MwtaLSTM, mwta, idx+1,train_data,valid_data,
                                                          test_data, dataset_settings, task_path)
    scores_folds.append(scores)