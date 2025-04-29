import torch
from torchvision.transforms import v2

from .customDataset import customDataset

def get_dataloaders(args, transformer, dataset_type):
    train_dataset = customDataset(
        mode="train",
        dataset_path=args.dataset,
        split_factor=args.split_factor,
        transformer=transformer,
        dataset_type=dataset_type
    )

    test_dataset = customDataset(
        mode="test",
        dataset_path=args.dataset,
        split_factor=args.split_factor,
        transformer=transformer,
        dataset_type=dataset_type
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False, # Não faz diferença nesse problema
        num_workers=4,
        pin_memory=False
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    return train_dataloader, test_dataloader

def get_transformer(grayscale, resize_size):
    return v2.Compose([
        v2.Resize((resize_size, resize_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=1) if grayscale else v2.Lambda(lambda x: x),
    ])

def get_dataset_type(dataset_path):
    if "bob_esponja_dataset" in dataset_path:
        return "bob_esponja"
    elif "imagenet" in dataset_path:
        return "tiny-imagenet"
    else:
        raise ValueError("Unknown dataset type. bob_esponja_dataset or tiny-imagenet expected.")
