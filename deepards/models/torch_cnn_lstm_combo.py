import torch
from torch.autograd import Variable
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNNLSTMNetwork(nn.Module):
    def __init__(self):
        super(CNNLSTMNetwork, self).__init__()

        # This is optimized for sequences of len 64 with 2 channels
        self.seq_size = 64

        # For now just stick with a normal NN. You can add skips later if this is successful
        self.breath_block = nn.Sequential(
            # first layer operates with SAME padding
            nn.Conv1d(2, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4),  # breaths should be down to about 16 points
            nn.Conv1d(64, 128, kernel_size=3, stride=1),  # 14 points
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, stride=1),  # 12 points
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, stride=1),  # 10 points
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=2, stride=1),  # 9 points
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=2, stride=1),  # 8 points
            nn.BatchNorm1d(256),
            nn.AvgPool1d(kernel_size=4),  # breaths should be down to 2 points
            Flatten(),  # should have 512 features
        )
        self.lstm_hidden_units = 512
        self.lstm_layers = 4
        self.lstm = nn.LSTM(512, self.lstm_hidden_units, num_layers=self.lstm_layers, batch_first=True)
        # lstm cell is silly because I need to implement the entire lstm chain
        # myself
        #self.lstm2 = nn.LSTMCell(512, 512)
        self.linear_final = nn.Linear(self.lstm_hidden_units, 5)
        self.softmax = nn.Softmax(dim=-1)
        self.flatten = Flatten()

    def init_hidden(self, batch_size):
        weight = next(self.lstm.parameters()).data
        return (Variable(weight.new(self.lstm_layers, batch_size, self.lstm_hidden_units).zero_()),
                Variable(weight.new(self.lstm_layers, batch_size, self.lstm_hidden_units).zero_()))

    def forward(self, x, hidden):
        # input should be in shape: (batches, breaths in seq, chans, 64)
        if x.shape[-1] != 64:
            raise Exception('input breaths must have sequence length of 64')
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.breath_block(x[0]).unsqueeze(dim=0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.breath_block(x[i]).unsqueeze(dim=0)], dim=0)
        x = self.lstm(outputs, hidden)[0]
        x = self.linear_final(x)
        return self.softmax(x)
