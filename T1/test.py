import torch
import sys
import torchvision
import os
import numpy as np
import albumentations as A
from PIL import Image
import random

from model import UNET
from customDataset import CustomDataset

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048
NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 1
NUM_CLASSES = 34

SAVE_IMAGES = True
N_IMAGES_TO_SAVE = 50

def main(leftImg8bit, gtFine, train_dir, device):
    global N_IMAGES_TO_SAVE
    device = f"cuda:{device}"

    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        A.pytorch.ToTensorV2()
    ])

    test_dataset = CustomDataset(
        image_folder=leftImg8bit + "val",
        mask_folder=gtFine + "val",
        transform=transform
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    model = UNET(in_channels=3, out_channels=NUM_CLASSES)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(train_dir, 'model.pth')))

    model.eval()

    total_pixels = 0
    correct_pixels = 0

    intersection = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)
    union = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)

    np.random.seed(0)
    palette = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)

    with torch.no_grad():
        for images, masks in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            preds = torch.argmax(model(images), dim=1)

            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            for i in range(NUM_CLASSES):
                pred_mask = (preds == i)
                true_mask = (masks == i)

                intersection_ = torch.logical_and(pred_mask, true_mask).sum().item()
                union_ = torch.logical_or(pred_mask, true_mask).sum().item()

                intersection[i] += intersection_
                union[i] += union_
            
            if SAVE_IMAGES and N_IMAGES_TO_SAVE > 0:
                img_vis = (images[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)  # [H, W, 3]

                mask_np = masks[0].cpu().numpy().astype(np.uint8)
                pred_np = preds[0].cpu().numpy().astype(np.uint8)

                mask_color = palette[mask_np]
                pred_color = palette[pred_np]

                concat = np.concatenate([img_vis, mask_color, pred_color], axis=1)

                img_out = Image.fromarray(concat)
                os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
                img_out.save(os.path.join(train_dir + "images", f"image_{N_IMAGES_TO_SAVE:05d}.png"))

                N_IMAGES_TO_SAVE -= 1

    pixel_accuracy = correct_pixels / total_pixels

    iou_per_class = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)
    for i in range(NUM_CLASSES):
        if union[i] > 0:
            iou_per_class[i] = intersection[i] / union[i]
        else:
            iou_per_class[i] = torch.tensor(0.0, device=device)
    
    mask_union_pos = (union > 0)
    if mask_union_pos.sum() > 0:
        mean_iou = iou_per_class[mask_union_pos].mean().item()
    else:
        mean_iou = 0.0

    with open(os.path.join(train_dir, "results.txt"), "w") as f:
        f.write(f"Pixel Accuracy: {pixel_accuracy:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write("IoU per class:\n")
        for i in range(NUM_CLASSES):
            f.write(f"Class {i}: {iou_per_class[i].item():.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python train.py <leftImg8bit> <gtFine> <train_dir> <device>")
        sys.exit(1)
    
    leftImg8bit = sys.argv[1]
    gtFine = sys.argv[2]
    train_dir = sys.argv[3]
    device = sys.argv[4]

    leftImg8bit += "/" if not leftImg8bit.endswith('/') else ""
    gtFine += "/" if not gtFine.endswith('/') else ""
    train_dir += "/" if not train_dir.endswith('/') else ""

    if not os.path.exists(train_dir):
        print(f"Training directory {train_dir} does not exist.")
        exit(1)

    main(leftImg8bit, gtFine, train_dir, device)
