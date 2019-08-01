import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(-1,target.long())
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1).long())
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


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
