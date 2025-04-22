from torch.utils.data import Dataset
from glob import glob
import os
from math import ceil
from PIL import Image

bob_esponja_class_map = {
    "bob_esponja": 0,
    "garry": 1,
    "larry": 2,
    "lula_molusco": 3,
    "patrick": 4,
    "plankton": 5,
    "sandy": 6,
    "siriguejo": 7,
}

'''
É importante que o dataset esteja balanceado
'''
class customDataset(Dataset):
    def __init__(self, mode, dataset_path, split_factor, transformer, dataset_type):
        self.images_paths = []
        self.targets = []

        self.mode = mode
        self.dataset_path = dataset_path
        self.split_factor = split_factor
        self.transformer = transformer
        self.dataset_type = dataset_type

        match dataset_type:
            case "bob_esponja":
                images_paths_ = sorted(glob(f"{dataset_path}/**/*.JPEG", recursive=True))
                targets_ = [bob_esponja_class_map[str(image.split("/")[-2])] for image in images_paths_]

                for class_name in os.listdir(dataset_path):
                    n_images = len(os.listdir(f"{dataset_path}{class_name}"))
                    images_split_factor = ceil(n_images * split_factor)

                    class_images_paths = sorted(glob(f"{dataset_path}/{class_name}/*.JPEG", recursive=True))
                    class_targets = [bob_esponja_class_map[class_name]] * len(class_images_paths)

                    if self.mode == "train":
                        self.images_paths += class_images_paths[:images_split_factor]
                        self.targets += class_targets[:images_split_factor]
                    elif self.mode == "test":
                        self.images_paths += class_images_paths[images_split_factor:]
                        self.targets += class_targets[images_split_factor:]

            case "tiny-imagenet":
                if mode == "train":
                    self.images_paths = glob(f"{dataset_path}/**/*.JPEG", recursive=True)
                    class_names = sorted(set(img.split("/")[-3] for img in self.images_paths))
                    self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
                    self.targets = [self.class_to_idx[img.split("/")[-3]] for img in self.images_paths]

                elif mode == "test":
                    with open(f"{dataset_path}val_annotations.txt", "r") as f:
                        lines = f.readlines()

                    image_name_to_class = {}
                    for line in lines:
                        l = line.split("\t")
                        image_name_to_class[l[0]] = l[1]

                    class_names = sorted(set(image_name_to_class.values()))
                    self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
                    
                    self.images_paths = glob(f"{dataset_path}/**/*.JPEG", recursive=True)
                    self.targets = [self.class_to_idx[image_name_to_class[img.split("/")[-1]]] for img in self.images_paths]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        target = self.targets[idx]
        image = self.transformer(Image.open(image_path))

        return image, target, image_path
