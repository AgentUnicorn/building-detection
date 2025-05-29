import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)

        # Dice loss calculation
        smooth = 1e-6
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        if targets.sum() == 0:
            return bce_loss

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )

        return bce_loss + dice_loss
