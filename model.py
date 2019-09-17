import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#import ipdb
from config import *


############### Embedding ######################

class EmbedModule(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(EmbedModule, self).__init__()

        self.input_dim =input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input: Batch x Temporal x Feature Dim
        # Batch x 512 x 1024
        x = x.permute(0, 2, 1)
        x = self.dropout(x.unsqueeze(3)).squeeze(3)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = self.relu(x)

        return x


########### GRU ####################

class GRUNet(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(GRUNet, self).__init__()

        self.hidden_dim = 32
        self.input_dim = input_dim
        self.base = EmbedModule(input_dim, 64, dropout_rate)
        self.middle = nn.GRU(
            input_size=64,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.phase_pred_net = nn.Sequential(
                nn.Linear(self.hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 7)
                )
        self.instrument_pred_net = nn.Sequential(
                nn.Linear(self.hidden_dim, 24),
                nn.ReLU(),
                nn.Linear(24, 21)
                )
        self.action_pred_net = nn.Sequential(
                nn.Linear(self.hidden_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 4)
                )

    def forward(self, x):
        x = self.base(x)
        batch_size = x.shape[0]
        frame_num = x.shape[1]

        h_state = torch.zeros(1, batch_size,self.hidden_dim).cuda().float()
        x, _ = self.middle(x, h_state)

        upsample_net = nn.UpsamplingBilinear2d([i3d_time*frame_num,
                                                self.hidden_dim])
        x = x.unsqueeze(0)
        x = upsample_net(x)
        x = x.squeeze(0)
        pred_phase = self.phase_pred_net(x)
        pred_instrument = self.instrument_pred_net(x)
        pred_action = self.action_pred_net(x)

        return pred_phase, pred_instrument, pred_action


# input_x = torch.randn(3, 512, 1024)
# model = GRUNet(1024)
# pred_phase, pred_instrument, pred_action = model(input_x)
# print(pred_phase[0].shape)
# print(pred_instrument.shape)
# print(pred_action.shape)


####################### TCN #################################

class TCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TCNEncoder, self).__init__()

        self.conv = nn.Conv1d(input_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class TCNDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TCNDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.ConvTranspose1d(input_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class TCNNet(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(TCNNet, self).__init__()

        self.base = EmbedModule(input_dim, 256, dropout_rate)

        self.middle = nn.Sequential(
            TCNEncoder(256, 64),
            TCNEncoder(64, 16),
            TCNDecoder(16, 64),
            TCNDecoder(64, 256),
            TCNDecoder(256, 512),
            TCNDecoder(512, 1024)
        )
        
        self.branch_size = 128

        self.phase_branch = nn.Sequential(
            nn.Linear(self.branch_size, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

        self.instrument_branch = nn.Sequential(
            nn.Linear(self.branch_size, 64),
            nn.ReLU(),
            nn.Linear(64, 21)
        )

        self.action_branch = nn.Sequential(
            nn.Linear(self.branch_size, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        frame_num = x.shape[1]
        x = self.base(x)

        padding = 4 - (x.shape[1] % 4)
        padding = padding % 4
        if padding != 0:
            x = nn.functional.pad(x, (0, 0, 0, padding), mode='constant',
                                      value=0)
        assert (x.shape[1] % 4 == 0)

        x = x.permute(0, 2, 1)
        x = self.middle(x)
        x = x.permute(0, 2, 1)

        if padding != 0:
            x = x[:, :-padding, :]    
        
        upsample_net = nn.UpsamplingBilinear2d([i3d_time * frame_num, self.branch_size])

        x = x.unsqueeze(0)
        x = upsample_net(x)
        x = x.squeeze(0)
        
        phase = self.phase_branch(x)
        instrument = self.instrument_branch(x)
        action = self.action_branch(x)
        
        return phase, instrument, action


####################### TCN_online #################################

class TCNonlineNet(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(TCNonlineNet, self).__init__()

        self.base = EmbedModule(input_dim, 256, dropout_rate)

        self.middle = nn.Sequential(
            TCNDecoder(512, 1024)
        )
        
        self.branch_size = 128

        self.phase_branch = nn.Sequential(
            nn.Linear(self.branch_size, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

        self.instrument_branch = nn.Sequential(
            nn.Linear(self.branch_size, 64),
            nn.ReLU(),
            nn.Linear(64, 21)
        )

        self.action_branch = nn.Sequential(
            nn.Linear(self.branch_size, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        frame_num = x.shape[1]
        x = self.base(x)

        padding = 4 - (x.shape[1] % 4)
        padding = padding % 4
        if padding != 0:
            x = nn.functional.pad(x, (0, 0, 0, padding), mode='constant',
                                      value=0)
        assert (x.shape[1] % 4 == 0)

        x = x.permute(0, 2, 1)
        x = self.middle(x)
        x = x.permute(0, 2, 1)

        if padding != 0:
            x = x[:, :-padding, :]    
        
        upsample_net = nn.UpsamplingBilinear2d([i3d_time * frame_num, self.branch_size])

        x = x.unsqueeze(0)
        x = upsample_net(x)
        x = x.squeeze(0)
        
        phase = self.phase_branch(x)
        instrument = self.instrument_branch(x)
        action = self.action_branch(x)
        
        return phase, instrument, action
              
