from torchvision import transforms


class ModelConstants:
    BATCH_SIZE = 32
    N_CHANNELS = 3  # RGB images
    N_CLASSES = 1
    NUM_EPOCHS = 94
    LEARNING_RATE = 0.001


class ImagesTransforms:
    IMAGE_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    MASK_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ]
    )
