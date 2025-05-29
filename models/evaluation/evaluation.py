def dice_coef(preds, targets, smooth=1e-7):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def iou_score(preds, targets, smooth=1e-7):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def f1_score(preds, targets, smooth=1e-7):
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    return 2 * (precision * recall) / (precision + recall + smooth)
