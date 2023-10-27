import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as  np

class FC_Encoder(nn.Module):
    def __init__(self, output_size):
        super(FC_Encoder, self).__init__()
        self.fc1 = nn.Linear(784, output_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return h1

class FC_Decoder(nn.Module):
    def __init__(self, embedding_size):
        super(FC_Decoder, self).__init__()
        self.fc3 = nn.Linear(embedding_size, 1024)
        self.fc4 = nn.Linear(1024, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(3, 64, 64)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,
                     out_channels=self.channel_mult*1,
                     kernel_size=3,
                     stride=1,
                     padding=1), # (B, C, H, W) --> (B, 16, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 3, 2, 1), # (B, 32, 32, 32)
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 3, 2, 1), # (B, 64, 16, 16)
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 3, 2, 1), # (B, 128, 8, 8)
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1), # (B, 256, 4, 4)
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv) # (B, 256*4*4)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size), # (B, output_size)  The input has become a one dimensional embedded vector 
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(3, 64, 64)):
        super(CNN_Decoder, self).__init__()
        self.input_height = 64
        self.input_width = 64
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = 3
        self.fc_output_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim), # (B, 512)
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*4, 8, 1, 0, bias=False), # (B, 64, 8, 8)
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 4, 2, 1, bias=False), # (B, 32, 16, 16)
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, 4, 2, 1, bias=False), # (B, 16, 32, 32)
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_mult*1, self.output_channels, 4, 2, 1, bias=False), # (B, 3, 64, 64)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x.view(-1, self.input_width*self.input_height)
