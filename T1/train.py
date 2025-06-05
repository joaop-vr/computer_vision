import torch
import sys
import torchvision
import os

from model import UNET
from customDataset import CustomDataset
import albumentations as A

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048
NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 1

def main(dataset_folder, output_dir, device):
    device = f"cuda:{device}"

    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        A.pytorch.ToTensorV2()
    ])

    train_dataset = CustomDataset(
        dataset_folder=dataset_folder,
        mode="train",
        transform=transform
    )

    val_dataset = CustomDataset(
        dataset_folder=dataset_folder,
        mode="val",
        transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    model = UNET(in_channels=3, out_channels=34)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_dataloader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            for images, masks in val_dataloader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks.long())
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train.py <dataset_folder> <output_dir> <device>")
        sys.exit(1)
    
    dataset_folder = sys.argv[1]
    output_dir = sys.argv[2]
    device = sys.argv[3]

    dataset_folder += "/" if not dataset_folder.endswith('/') else ""
    output_dir += "/" if not output_dir.endswith('/') else ""

    os.makedirs(output_dir, exist_ok=True)

    main(dataset_folder, output_dir, device)
