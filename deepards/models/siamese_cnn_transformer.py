import torch
from torch.autograd import Variable
import torch.nn as nn

from deepards.models.transformer import Transformer


class SiameseCNNTransformerNetwork(nn.Module):
    def __init__(self, breath_block, hidden_units, sub_batch_size):
        super(SiameseCNNTransformerNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        num_blocks = 2
        self.transformer = Transformer(breath_block.n_out_filters, hidden_units, num_blocks, 4)
        self.linear_intermediate = nn.Linear(hidden_units, 2)
        self.linear_final = nn.Linear(2 * sub_batch_size, 2)

    def _run_through_model(self, inputs):
        batches = inputs.shape[0]
        breaths = inputs.shape[1]
        outputs = self.breath_block(inputs[0]).squeeze().unsqueeze(dim=0)
        for i in range(1, batches):
            block_out = self.breath_block(inputs[i]).squeeze().unsqueeze(dim=0)
            outputs = torch.cat([outputs, block_out], dim=0)
        return self.transformer(outputs)

    def forward(self, x, compr):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        x_out = self._run_through_model(x)
        compr_out = self._run_through_model(compr)

        diff = self.linear_intermediate(torch.abs(compr_out - x_out))
        diff = diff.view((diff.shape[0], -1))
        return self.linear_final(diff)
