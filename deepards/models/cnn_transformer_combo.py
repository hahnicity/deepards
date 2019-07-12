import torch
from torch.autograd import Variable
import torch.nn as nn

from deepards.models.transformer import Transformer


class CNNTransformerNetwork(nn.Module):
    def __init__(self, breath_block, metadata_features, bm_to_linear, hidden_units, num_blocks):
        super(CNNTransformerNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.bm_to_linear = bm_to_linear

        if not bm_to_linear:
            self.transformer = Transformer(breath_block.n_out_filters+metadata_features, hidden_units, num_blocks, 3)
            self.linear_final = nn.Linear(self.lstm_hidden_units+metadata_features, 2)
        else:
            self.transformer = Transformer(breath_block.n_out_filters, hidden_units, num_blocks, 3)
            self.linear_final = nn.Linear(self.lstm_hidden_units+metadata_features, 2)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, metadata):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.breath_block(x[0]).squeeze()
        if not torch.any(torch.isnan(metadata)) and not self.bm_to_linear:
            outputs = torch.cat([outputs, metadata[0]], dim=-1)
        outputs = outputs.unsqueeze(dim=0)

        for i in range(1, batches):
            block_out = self.breath_block(x[i]).squeeze()
            if not torch.any(torch.isnan(metadata)) and not self.bm_to_linear:
                block_out = torch.cat([block_out, metadata[i]], dim=-1)
            block_out = block_out.unsqueeze(dim=0)
            outputs = torch.cat([outputs, block_out], dim=0)

        x = self.transformer(outputs)
        if self.bm_to_linear:
            x = torch.cat([x, metadata], dim=-1)
        x = self.linear_final(x)
        return self.softmax(x)
