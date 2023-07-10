import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.criterion = nn.BCELoss( reduction='none')

    def forward(self, outputs, targets):
        targets = targets.reshape(-1)
        mask = targets >= 0
        outputs = outputs.permute(0, 2, 3, 1).reshape((-1, 2))
        targets = targets[mask]
        outputs = outputs[mask.unsqueeze(-1).repeat(1, 2)].view(-1, 2)
        BCE_loss = self.criterion(outputs, targets)
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
