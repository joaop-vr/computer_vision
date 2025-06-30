import sys
import os
from glob import glob
import shutil

def main(leftImg8bit, gtFine, output_dir, n_cities_val, n_cities_test):
    images = {}
    for city in sorted(os.listdir(f"{leftImg8bit}train")):
        images[city] = sorted(glob(f"{leftImg8bit}train/{city}/**/*leftImg8bit*.png", recursive=True))
    for city in sorted(os.listdir(f"{leftImg8bit}val")):
        images[city] = sorted(glob(f"{leftImg8bit}val/{city}/**/*leftImg8bit*.png", recursive=True))

    labelIds = {}
    for city in sorted(os.listdir(f"{gtFine}train")):
        labelIds[city] = sorted(glob(f"{gtFine}train/{city}/**/*labelIds.png", recursive=True))
    for city in sorted(os.listdir(f"{gtFine}val")):
        labelIds[city] = sorted(glob(f"{gtFine}val/{city}/**/*labelIds.png", recursive=True))

    n_cities = len(images)
    n_cities_train = n_cities - n_cities_val - n_cities_test
    if n_cities_train <= 0:
        raise ValueError("n_cities_val or n_cities_test is too large, resulting in no training cities.")
    
    train_cities = list(images.keys())[:n_cities_train]
    val_cities = list(images.keys())[n_cities_train:n_cities_train + n_cities_val]
    test_cities = list(images.keys())[n_cities_train + n_cities_val:]

    create_folder(output_dir, "train", train_cities, images, labelIds)
    create_folder(output_dir, "val", val_cities, images, labelIds)
    create_folder(output_dir, "test", test_cities, images, labelIds)

def create_folder(output_dir, split, cities, images, labelIds):
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for city in cities:
        city_dir = os.path.join(split_dir, city)
        os.makedirs(city_dir, exist_ok=True)

        for img_path in images[city]:
            img_name = os.path.basename(img_path)
            new_img_path = os.path.join(city_dir, img_name)
            shutil.copy(img_path, new_img_path)

        for label_path in labelIds[city]:
            label_name = os.path.basename(label_path)
            new_label_path = os.path.join(city_dir, label_name)
            shutil.copy(label_path, new_label_path)

    print(f"Created {split} set with {len(cities)} cities in {split_dir}: {len(glob(os.path.join(split_dir, '**', '*leftImg8bit*.png'), recursive=True))} images")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python train.py <leftImg8bit> <gtFine> <output_dir> <n_cities_val> <n_cities_test>")
        sys.exit(1)
    
    leftImg8bit = sys.argv[1]
    gtFine = sys.argv[2]
    output_dir = sys.argv[3]
    n_cities_val = int(sys.argv[4])
    n_cities_test = int(sys.argv[5])

    leftImg8bit += "/" if not leftImg8bit.endswith('/') else ""
    gtFine += "/" if not gtFine.endswith('/') else ""
    output_dir += "/" if not output_dir.endswith('/') else ""

    os.makedirs(output_dir, exist_ok=True)

    main(leftImg8bit, gtFine, output_dir, n_cities_val, n_cities_test)
