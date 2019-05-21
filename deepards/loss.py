import torch
from torch import nn


class VacillatingLoss(nn.Module):
    def __init__(self, alpha):
        super(VacillatingLoss, self).__init__()
        self.left_hand_func = lambda x: -torch.log(2 * (torch.exp(-alpha) - 1) * x + 1)
        self.right_hand_func = lambda x: -torch.log(2 * torch.exp(-alpha) * (1 - x) + 2 * x - 1)
        self.bce = nn.BCELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        lh = self.left_hand_func((pred.sum(dim=1) / pred.shape[1]))
        rh = self.right_hand_func((pred.sum(dim=1) / pred.shape[1]))
        # ensure that nans and values above alpha are dealt with
        lh[(lh > self.alpha) | torch.isnan(lh)] = rh[rh <= self.alpha]
        vacillating_loss = lh.mean()
        return bce_loss + vacillating_loss
