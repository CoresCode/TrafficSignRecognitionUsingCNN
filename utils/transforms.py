import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 128


def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.GaussNoise(
            std_range=(0.02, 0.08),
            p=0.3
        ),

        A.MotionBlur(
            blur_limit=3,
            p=0.2
        ),

        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),

        A.Rotate(
            limit=10,
            border_mode=0,
            p=0.4
        ),

        A.Perspective(
            scale=(0.03, 0.08),
            p=0.25
        ),

        A.ToGray(
            p=0.15
        ),

        A.Normalize(),

        ToTensorV2()
    ])


def get_valid_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])


def get_test_transforms():
    return get_valid_transforms()