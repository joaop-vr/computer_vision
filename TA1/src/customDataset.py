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

                    if self.mode == "train":
                        self.images_paths += images_paths_[:images_split_factor]
                        self.targets += targets_[:images_split_factor]
                    elif self.mode == "test":
                        self.images_paths += images_paths_[images_split_factor:]
                        self.targets += targets_[images_split_factor:]

            case "imagenet":
                pass

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        target = self.targets[idx]
        image = self.transformer(Image.open(image_path))

        return image, target, image_path
