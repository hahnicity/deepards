import torch
from torch import nn


class VacillatingLoss(nn.Module):
    def __init__(self, alpha):
        super(VacillatingLoss, self).__init__()
        self.left_hand_func = lambda x: -torch.log(2 * (torch.exp(-alpha) - 1) * x + 1)
        self.right_hand_func = lambda x: -torch.log(2 * torch.exp(-alpha) * (1 - x) + 2 * x - 1)
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        vacillating_sum = 0
        for i in range(len(pred)):
            # XXX this removes the grad function.
            batch_n = (pred[i].argmax(dim=1).sum()) / len(pred[i])
            if batch_n > .5:
                vacillating_sum += self.right_hand_func(batch_n)
            else:
                vacillating_sum += self.left_hand_func(batch_n)
        vacillating_loss = vacillating_sum / len(pred)
        return bce_loss + vacillating_loss
