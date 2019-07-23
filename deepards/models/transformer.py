"""
transformer
~~~~~~~~~~~

Credit to https://github.com/jensjepsen/imdb-transformer/blob/master/model.py
with slight modifications and bug fixes by myself
"""
import torch
from torch import nn
from torch.autograd import Variable


class MultiHeadAttention(nn.Module):
    """
        A multihead attention module,
        using scaled dot-product attention.
    """
    def __init__(self,input_size,hidden_size,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.head_size = self.hidden_size / num_heads

        self.q_linear = nn.Linear(self.input_size,self.hidden_size)
        self.k_linear = nn.Linear(self.input_size,self.hidden_size)
        self.v_linear = nn.Linear(self.input_size,self.hidden_size)

        self.joint_linear = nn.Linear(self.hidden_size,self.input_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,q,k,v):
        # project the queries, keys and values by their respective weight matrices
        q_proj = self.q_linear(q).view(q.size(0), q.size(1), self.num_heads,self.head_size).transpose(1,2)
        k_proj = self.k_linear(k).view(k.size(0), k.size(1), self.num_heads,self.head_size).transpose(1,2)
        v_proj = self.v_linear(v).view(v.size(0), v.size(1), self.num_heads,self.head_size).transpose(1,2)


        # calculate attention weights
        unscaled_weights = torch.matmul(q_proj,k_proj.transpose(2,3))
        weights = self.softmax(unscaled_weights / torch.sqrt(torch.Tensor([self.head_size * 1.0]).to(unscaled_weights)))

        # weight values by their corresponding attention weights
        weighted_v = torch.matmul(weights,v_proj)
        weighted_v = weighted_v.transpose(1,2).contiguous()

        # do a linear projection of the weighted sums of values
        joint_proj = self.joint_linear(weighted_v.view(q.size(0),q.size(1),self.hidden_size))

        # store a reference to attention weights, for THIS forward pass,
        # for visualisation purposes
        self.weights = weights

        return joint_proj


class Block(nn.Module):
    """
        One block of the transformer.
        Contains a multihead attention sublayer
        followed by a feed forward network.
    """
    def __init__(self, input_size, hidden_size, num_heads, activation, dropout):
        super(Block,self).__init__()
        self.dropout = dropout

        self.attention = MultiHeadAttention(input_size, hidden_size, num_heads)
        self.attention_norm = nn.LayerNorm(input_size)

        ff_layers = [
            nn.Linear(input_size,hidden_size),
            activation(),
            nn.Linear(hidden_size,input_size),
            ]

        self.attention_dropout = nn.Dropout(dropout)
        ff_layers.append(nn.Dropout(dropout))

        self.ff = nn.Sequential(
            *ff_layers
            )
        self.ff_norm = nn.LayerNorm(input_size)

    def forward(self,x):
        attended = self.attention_norm(self.attention_dropout(self.attention(x,x,x)) + x)
        return self.ff_norm(self.ff(attended) + x)


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, num_heads, activation=nn.ReLU, dropout=.2):
        """
            A single Transformer Network
        """
        super(Transformer,self).__init__()

        self.blocks = nn.Sequential(
                *[Block(input_size,hidden_size,num_heads,activation,dropout=dropout)
                    for _ in xrange(num_blocks)]
            )

    def forward(self,x):
        """
            Sequentially applies the blocks of the Transformer network
        """
        return self.blocks(x)


if __name__ == "__main__":
    """
        If run seperately, does a simple sanity check,
        by doing a random forward pass
    """
    t = Transformer(512, 512, 5, 4)

    input = Variable(torch.rand(16,100,512))

    print t(input)
