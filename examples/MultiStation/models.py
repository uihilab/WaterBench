import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Benchmark LSTM

    Parameters:
        input_dim, int:
            Input dimension
        hidden_dim, int:
            Latent dimension
        output_dim, int:
            Output dimension
        num_layers, int:
            Number of LSTM layers
        dropout, float:
            Dropout value, Default is ``0``.
        bidirectional, bool:
            If ``True``, becomes a bidirectional LSTM. Default: ``False``
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0, bidirectional=False, **kwargs):
        super().__init__(**kwargs)

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        if bidirectional:
            hidden_dim *= 2
        self.linear = nn.Linear(hidden_dim, output_dim)

        self.name = 'LSTM'

    def forward(self, x):
        """
        Propagate input through the network

        Parameters:
            x: torch.Tensor with shape (m, K, input_dim)

        Returns:
            torch.Tensor with shape (m, K, output_dim)
        """
        rnn_out, _ = self.rnn(x)
        output = self.linear(rnn_out)
        return output


class BiGRU(LSTM):
    """
    Benchmark Bidirictionnal GRU

    Parameters:
        input_dim, int:
            Input dimension
        hidden_dim, int:
            Latent dimension
        output_dim, int:
            Output dimension
        num_layers, int:
            Number of LSTM layers
        dropout, float:
            Dropout value, Default is ``0``
        bidirectional, bool:
            If ``True``, becomes a bidirectional GRU. Default: ``True``
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0, bidirectional=False, **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional, **kwargs)

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        self.name = 'GRU'


class ConvGru(nn.Module):
    """
    Benchmark Convolutional GRU

    Parameters:
        input_dim, int:
            Input dimension
        hidden_dim, int:
            Latent dimension
        output_dim, int:
            Output dimension
        num_layers, int:
            Number of LSTM layers
        dropout, float:
            Dropout value, Default is ``0``
        bidirectional, bool:
            If ``True``, becomes a bidirectional GRU. Default: ``False``
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0, bidirectional=False, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=11, stride=1, padding=11//2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=11, stride=1, padding=11//2)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=11, stride=1, padding=11//2)

        self.activation = nn.LeakyReLU(0.1)

        self.rnn = BiGRU(hidden_dim,
                         hidden_dim,
                         output_dim,
                         num_layers,
                         dropout=dropout,
                         bidirectional=bidirectional)

        self.name = 'ConvGru'

    def forward(self, x):
        """
        Propagate input through the network

        Parameters:
            x: torch.Tensor with shape (m, K, input_dim)

        Returns:
            torch.Tensor with shape (m, K, output_dim)
        """
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = x.transpose(1, 2)

        x = self.rnn(x)

        return x


class FullyConv(nn.Module):
    """
    Benchmark ConvNet

    Parameters:
        input_dim, int:
            Input dimension
        hidden_dim, int:
            Latent dimension
        output_dim, int:
            Output dimension
        dropout, float:
            Dropout value, Default is ``0``
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=11, stride=1, padding=11//2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=11, stride=1, padding=11//2)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=11, stride=1, padding=11//2)

        self.activation = nn.LeakyReLU(0.1)

        self.name = 'FullyConv'

    def forward(self, x):
        """
        Propagate input through the network

        Parameters:
            x: torch.Tensor with shape (m, K, input_dim)

        Returns:
            torch.Tensor with shape (m, K, output_dim)
        """
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = x.transpose(1, 2)

        return x


class FFN(nn.Module):
    """
    Benchmark FFN

    Parameters:
        input_dim, int:
            Input dimension
        hidden_dim, int:
            Latent dimension
        output_dim, int:
            Output dimension
        num_layers, int:
            Number of layers
        dropout, float:
            Dropout value, Default is ``0``
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)

        layer_dim = [hidden_dim for _ in range(num_layers)]
        layer_dim[0] = input_dim
        layer_dim[-1] = output_dim

        self.layers_dense = nn.ModuleList([nn.Linear(layer_dim[i],
                                                     layer_dim[i+1]) for i in range(num_layers-1)])

        self.name = 'FFN'

    def forward(self, x):
        """
        Propagate input through the network

        Parameters:
            x: torch.Tensor with shape (m, K, input_dim)

        Returns:
            torch.Tensor with shape (m, K, output_dim)
        """
        for layer in self.layers_dense:
            x = layer(x)
        return x