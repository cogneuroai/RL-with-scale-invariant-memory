# define the attention layer

# feed forward
# qkv to attention calculation
# attention layer
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, n_heads):
        """
        multi head attention using scaled dot product according to "Attention is all we need"
        :param input_dim: input dimension for q, k, and v: a list
        :param d_model: model dimension, output dim of Wq, Wk, and Wv
        :param n_heads: number of heads
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, " dimension of the model must be divisible by number of heads"

        self.d_model = d_model
        self.n_heads = n_heads
        # modify the d values below if we want to have more flexibility when defining the attention module\
        self.d_q = input_dim[0]
        self.d_k = input_dim[1]
        self.d_v = input_dim[2]
        self.w_q = nn.Linear(self.d_q, self.d_model)
        self.w_k = nn.Linear(self.d_k, self.d_model)
        self.w_v = nn.Linear(self.d_v, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, in_q, in_k, in_v, mask=None):
        """
        forward call
        :param in_q: has dim (N, L, d_model)
        :param in_k: has dim (N, L, d_model)
        :param in_v: has dim (N, L, d_model)
        :param mask: masking attention
        :return: attention weight @ value
        """
        b, t, d = in_q.shape
        _, s, _ = in_k.shape
        # get the q,k,v
        q = self.w_q(in_q)  # has dim b, t, d_model
        k = self.w_k(in_k)  # has dim d, t, d_model
        v = self.w_v(in_v)  # has dim d, t, d_model
        # print(q.shape, k.shape, v.shape)
        # reshaping to split into multiple heads before feeding into scaled dot product attention
        q = q.view(b, t, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        z = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # dim: b, n_heads,t, d_model
        # merging nhead and Ev dimension
        z = torch.flatten(z.transpose(1, 2), -2)  # dim b, t, d_model
        output = self.w_o(z)

        return output


class PositionWiseFFNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFFNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.ffl1 = nn.Linear(d_model, d_ff)
        self.ffl2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.ffl2(self.relu(self.ffl1(x)))

        return output


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, d_ff, dropout):
        """
        an attention layer
        :param input_dim: dimension of the input for q, k, and v: a list
        :param output_dim: dimension of the output
        :param n_heads: number of heads
        :param d_ff: intermediate layer node in position-wise feed forward network
        :param dropout: dropout for skip connection
        """
        super(AttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(input_dim, output_dim, n_heads)
        self.ffnet = PositionWiseFFNet(output_dim, d_ff)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tilde_f, mask=None):
        """

        :param x: have dim (N, 1, f)
        :param tilde_f: (N,1,f,tau)
        :param mask:
        :return:
        """
        # concatenate tilde_f with x for k and v
        k_v = torch.cat((x.unsqueeze(-1), tilde_f), -1)[:, 0, :, :].transpose(1, 2)  # dim: b, tau+1, f
        x_attn_output = self.attention(x, k_v, k_v, mask)
        x = self.layer_norm1(x + self.dropout(x_attn_output))
        ff_output = self.ffnet(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x