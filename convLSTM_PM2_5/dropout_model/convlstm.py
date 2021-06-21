import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_consts as cnst
from load_data import load_bj_aqi_station_locations


class ConvLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            input_channel,
            hidden_channel,
            kernel_size,
            stride=1,
            padding=0):
        """
        Initializations

        :param input_size: (int, int): height, width tuple of the input
        :param input_channel: int: number of channels of the input
        :param hidden_channel: int: number of channels of the hidden state
        :param kernel_size: int: size of the filter
        :param stride: int: stride
        :param padding: int: width of the 0 padding
        """

        super(ConvLSTM, self).__init__()
        self.n_h, self.n_w = input_size
        self.n_c = input_channel
        self.hidden_channel = hidden_channel

        self.conv_xi = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xf = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xo = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xg = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hi = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hf = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_ho = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hg = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hi = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, hidden_states):
        """
        Forward prop.

        :param x: input tensor of shape (n_batch, n_c, n_h, n_w)
        :param hidden_states: (tensor, tensor) for hidden and cell states.
                              Each of shape (n_batch, n_hc, n_hh, n_hw)
        :return: (hidden_state, cell_state)
        """

        hidden_state, cell_state = hidden_states

        xi = self.conv_xi(x)
        hi = self.conv_hi(hidden_state)
        xf = self.conv_xf(x)
        hf = self.conv_hf(hidden_state)
        xo = self.conv_xo(x)
        ho = self.conv_ho(hidden_state)
        xg = self.conv_xg(x)
        hg = self.conv_hg(hidden_state)

        i = torch.sigmoid(xi + hi)
        f = torch.sigmoid(xf + hf)
        o = torch.sigmoid(xo + ho)
        g = torch.tanh(xg + hg)

        cell_state = f * cell_state + i * g
        hidden_state = o * torch.tanh(cell_state)

        return hidden_state, cell_state

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda(),
                torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda())


class ConvLSTMForecast2L(nn.Module):
    def __init__(self, input_size, hidden_dim, kernel_size, padding):
        super(ConvLSTMForecast2L, self).__init__()

        self.convlstm1 = ConvLSTM(
            input_size,
            6,
            hidden_dim,  # 128
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.convlstm2 = ConvLSTM(
            input_size,
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        (self.hidden1, self.cell1), (self.hidden2, self.cell2) = self.init_hidden(10)

        self.meo_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.meo_conv_output = nn.Conv2d(hidden_dim, 6, 1)  # The final meo prediction layer

    def forward(self, X, hidden_states=None):
        m, Tx, n_c, n_h, n_w = X.shape
        dropout = nn.Dropout(p=0.05)
        meo_output = []

        if hidden_states:
            (self.hidden1, self.cell1), (self.hidden2, self.cell2) = hidden_states
        else:
            (self.hidden1, self.cell1), (self.hidden2, self.cell2) = self.init_hidden(m)

        for t in range(Tx):
            xt = X[:, t, :, :, :]
            self.hidden1, self.cell1 = self.convlstm1(xt, (self.hidden1, self.cell1))
            self.hidden1, self.cell1 = dropout(self.hidden1), dropout(self.cell1)
            self.hidden2, self.cell2 = self.convlstm2(self.hidden1, (self.hidden2, self.cell2))
            self.hidden2, self.cell2 = dropout(self.hidden2), dropout(self.cell2)

            # MEO prediction
            meo_pred = torch.tanh(self.meo_conv_1(self.hidden2))
            meo_pred = self.meo_conv_output(meo_pred)
            meo_pred = meo_pred.view(m, 1, 6, n_h, n_w)
            meo_output.append(meo_pred)

        # cat on time dimension
        meo_output = torch.cat(meo_output, dim=1)
        hidden_states = (self.hidden1, self.cell1), (self.hidden2, self.cell2)

        return meo_output, hidden_states

    def init_hidden(self, batch_size):
        return self.convlstm1.init_hidden(batch_size), \
               self.convlstm2.init_hidden(batch_size)
