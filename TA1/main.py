import os
import argparse
import yaml
import cv2
import numpy as np
import json

from src.utils import get_dataloaders, get_transformer, get_dataset_type

def main():
    args = get_args()
    handle_args(args)
    save_args(args)
    dataset_type = get_dataset_type(args.dataset)

    filters_config = yaml.safe_load(open(args.filters_config, "r"))

    transformer = get_transformer(args.grayscale, args.resize_size)

    train_dataloader, test_dataloader = get_dataloaders(args, transformer, dataset_type)

    train_features = extract_features(train_dataloader, filters_config, args.window_size)
    test_features = extract_features(test_dataloader, filters_config, args.window_size)

    save_features(train_features, args.output + "train_features.json")
    save_features(test_features, args.output + "test_features.json")

'''
Devolve um dicionário.....................................................
'''
def extract_features(dataloader, filters_config, window_size):
    features = {}

    for image, label, image_path in dataloader:
        for filter_name in filters_config["filtros"]:
            kernels = filters_config[filter_name]["kernels"]

            for kernel in kernels:
                filtro = filters_config[filter_name]["filtros"][f"kernel_{str(kernel)}"]

                filtro_features = apply_filter(image, kernel, filtro, window_size)

                append_on_dict(features, image_path[0], filter_name, kernel, filtro_features, label[0])
    
    return features

def append_on_dict(features, image_path, filter_name, kernel, filtro_features, label):
    if int(kernel) not in features:
        features[int(kernel)] = {}

    if str(filter_name) not in features[int(kernel)]:
        features[int(kernel)][str(filter_name)] = {}

    features[int(kernel)][str(filter_name)][str(image_path)] = {}
    features[int(kernel)][str(filter_name)][str(image_path)]["features"] = np.array(filtro_features, dtype=np.float32).tolist()
    features[int(kernel)][str(filter_name)][str(image_path)]["label"] = int(label)

    return features

def apply_filter(image, kernel, filtro, window_size):
    kernel = np.array(filtro, dtype=np.float32)
    image = image.squeeze(0).numpy()

    filtered_image = cv2.filter2D(image, -1, kernel)

    janelas = divide_image(filtered_image, window_size)

    return [np.mean(janela) for janela in janelas]

def divide_image(image, window_size):
    height, width = image.shape[1:]
    janelas = []

    for i in range(0, height, window_size):
        for j in range(0, width, window_size):
            janela = image[:, i:i + window_size, j:j + window_size]
            
            janelas.append(janela)

    return janelas

def save_features(features, path):
    with open(path, "w") as f:
        json.dump(features, f)

def save_args(args):
    dict = {}

    dict["dataset"] = args.dataset
    dict["output"] = args.output
    dict["split_factor"] = args.split_factor
    dict["grayscale"] = args.grayscale
    dict["filters_config"] = args.filters_config
    dict["window_size"] = args.window_size

    json.dump(dict, open(args.output + "args.json", "w"))

def handle_args(args):
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset path {args.dataset} does not exist.")
    
    if args.split_factor <= 0 or args.split_factor >= 1:
        raise ValueError("split_factor must be between 0 and 1.")
    
    if args.grayscale.lower() not in ["true", "false"]:
        raise ValueError("grayscale must be 'True' or 'False'.")
    
    if not os.path.exists(args.filters_config):
        raise FileNotFoundError(f"Filters configuration file {args.filters_config} does not exist.")
    
    args.dataset += "/" if not args.dataset.endswith("/") else ""
    args.output += "/" if not args.output.endswith("/") else ""

    args.grayscale = True if args.grayscale.lower() == "true" else False

    if args.resize_size % args.window_size != 0:
        raise ValueError("resize_size must be divisible by window_size.")

    os.makedirs(args.output, exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser(description="Segmentation script")
    
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--split_factor", type=float, required=True, help="Factor to split the dataset into train and test")
    parser.add_argument("--grayscale", type=str, required=True, help="Whether to convert images to grayscale (True/False)")
    parser.add_argument("--filters_config", type=str, required=True, help="Path to the filters configuration file")
    parser.add_argument("--resize_size", type=int, required=True, help="Size to resize images")
    parser.add_argument("--window_size", type=int, required=True, help="Size of the window for filtering")

    return parser.parse_args()

if __name__ == "__main__":
    main()