import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision.ops.focal_loss import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if not size_average:
            self.reduction = 'sum'
        else:
            self.reduction = 'mean'

    def forward(self, input, target):
        # dont do any transform on input because the sigmoid_focal_loss func
        # is performing F.binary_cross_entropy_with_logits
        return sigmoid_focal_loss(input, target, reduction=self.reduction)


class VacillatingLoss(nn.Module):
    def __init__(self, alpha):
        super(VacillatingLoss, self).__init__()
        self.left_hand_func = lambda x: -torch.log(2 * (torch.exp(-alpha) - 1) * x + 1)
        self.right_hand_func = lambda x: -torch.log(2 * torch.exp(-alpha) * (1 - x) + 2 * x - 1)
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred = nn.Softmax(dim=-1)(pred)
        lh = self.left_hand_func((pred.sum(dim=1) / pred.shape[1]))
        rh = self.right_hand_func((pred.sum(dim=1) / pred.shape[1]))
        # ensure that nans and values above alpha are dealt with
        lh[(lh > self.alpha) | torch.isnan(lh)] = rh[rh <= self.alpha]
        vacillating_loss = lh.mean()
        return bce_loss + vacillating_loss


class ConfidencePenaltyLoss(nn.Module):
    def __init__(self, beta):
        super(ConfidencePenaltyLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        confidence_loss = -(self.beta * (F.softmax(pred, dim=-1) * F.log_softmax(pred, dim=-1))).mean()
        return (bce_loss - confidence_loss)
