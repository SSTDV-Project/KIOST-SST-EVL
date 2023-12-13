import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def FrechetGEVL(pred, target, a=13, s=1.7):
    temp = torch.abs(pred - target) / s + np.power(a / (1 + a), 1 / a)
    return (0 * (-1 - a) * (-temp ** -a) + torch.log(temp)).mean()

def GumbelGEVL(pred, target, r=1.0):
    return (torch.pow(1 - torch.exp(-torch.pow(pred - target, 2)), r) * torch.pow(pred - target, 2)).mean()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_sequence_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class SineActivation(nn.Module):
    def __init__(self, channels):
        super(SineActivation, self).__init__()

        self.fc = TimeDistributed(nn.Linear(1, channels[1]), batch_first=True)

    def forward(self, x):
        y = None

        for i in range(x.size(2)):
            f = self.fc(x[:, :, i:i + 1])
            f = torch.cat([torch.sin(f[:, :, :-1]), f[:, :, -1:]], dim=-1)
            y = f if y is None else torch.cat([y, f], dim=-1)

        return y

class TimeDistributed(nn.Module):
    # takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    # from: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1)) # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # we have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1)) # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1)) # (timesteps, samples, output_size)

        return y

class Transformer(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, time_step):
        super(Transformer, self).__init__()

        self.forward1 = nn.Sequential(
                TimeDistributed(nn.Linear(hidden_size, hidden_size)),
                TimeDistributed(nn.BatchNorm1d(hidden_size)),
                nn.ReLU(inplace=True),
                TimeDistributed(nn.Linear(hidden_size, output_size)))
        
        self.fc = TimeDistributed(nn.Linear(1 + 2 * hidden_size + 16, hidden_size))
        self.positional_encoding = PositionalEncoding(hidden_size, max_sequence_length=(time_step[0] + time_step[1]))
        self.sa = SineActivation([2, hidden_size])
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=hidden_size, dropout=0)

    def forward(self, static_variable, encoder_y, encoder_x, decoder_y, decoder_x=None):
        z = static_variable.unsqueeze(1)
        
        x = torch.cat([encoder_x, self.sa(encoder_y), z.expand(z.size(0), encoder_y.size(1), z.size(2))], dim=2)    # batch_size, time_step, channel
        x = x.permute(1, 0, 2)                                                                                      # time_step, batch_size, channel
        x = self.fc(x)

        if decoder_x is None:
            decoder_x = torch.zeros_like(encoder_x)

        y = torch.cat([decoder_x, self.sa(decoder_y), z.expand(z.size(0), decoder_y.size(1), z.size(2))], dim=2)
        y = y.permute(1, 0, 2)
        y = self.fc(y)

        z = torch.cat([x, y], dim=0)
        z = self.positional_encoding(z)
        
        x = self.transformer(z[:x.size(0)], z[x.size(0):])
        x = self.forward1(x)
        x = x.transpose(0, 1)
        
        return x

        '''
        o = []

        m = self.transformer_encoder(x)
        x = self.forward1(m)

        for i in range(decoder_y.size(1)):            
            y = torch.cat([x[-1], decoder_y[:, i], z[:, 0]], dim=1).unsqueeze(0)
            y = self.fc1(y)
            y = self.transformer_decoder(y, m)
            x = self.forward1(y)
            o.append(x)

        return torch.cat(o).transpose(0, 1)
        '''
