import albumentations as A
from albumentations.pytorch import ToTensorV2


def getValidTransform():
    return A.Compose([A.Resize(448, 448), A.Normalize(), ToTensorV2()])


def getTrainTransform():
    return A.Compose(
        [
            A.Resize(448, 448),  # Make divisible by 32 (for 5-depth UNet)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
