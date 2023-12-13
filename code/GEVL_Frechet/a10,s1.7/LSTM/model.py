import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def FrechetGEVL(pred, target, a=10, s=1.7):
    temp = torch.abs(pred - target) / s + np.power(a / (1 + a), 1 / a)
    return ((temp ** -a) + (1 + a) * torch.log(temp)).mean()

def GumbelGEVL(pred, target, r=1.1):
    return (torch.pow(1 - torch.exp(-torch.pow(pred - target, 2)), r) * torch.pow(pred - target, 2)).mean()

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

class LSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()

        self.forward1 = nn.Sequential(
                TimeDistributed(nn.Linear(hidden_size, hidden_size)),
                TimeDistributed(nn.BatchNorm1d(hidden_size)),
                nn.ReLU(inplace=True),
                TimeDistributed(nn.Linear(hidden_size, output_size)))

        self.fc   = TimeDistributed(nn.Linear(1 + 2 * hidden_size + 16, hidden_size))
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        self.sa   = SineActivation([2, hidden_size])

    def forward(self, static_variable, encoder_y, encoder_x, decoder_y, decoder_x=None):
        z = static_variable.unsqueeze(1)
        
        x = torch.cat([encoder_x, self.sa(encoder_y), z.expand(z.size(0), encoder_y.size(1), z.size(2))], dim=2) # batch_size, time_step, channel
        x = x.permute(1, 0, 2)                                                                          # time_step, batch_size, channel
        x = self.fc(x)
        x, (h, s) = self.lstm(x)

        if decoder_x is None:
            decoder_x = torch.zeros_like(encoder_x)

        x = torch.cat([decoder_x, self.sa(decoder_y), z.expand(z.size(0), decoder_y.size(1), z.size(2))], dim=2)
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        x = self.lstm(x, (h, s))[0]
        x = self.forward1(x).transpose(0, 1)

        return x
