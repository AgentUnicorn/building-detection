import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        # Dice loss calculation
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        
        return bce_loss + dice_loss
