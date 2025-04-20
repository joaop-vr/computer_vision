import os
import json
import argparse
import numpy as np
from scipy.spatial.distance import euclidean
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    args = get_args()
    handle_args(args)

    train_features_json = json.load(open(args.features_dir + "train_features.json", "r"))
    test_features_json = json.load(open(args.features_dir + "test_features.json", "r"))

    n_windows = get_n_windows(args.features_dir)

    classify_images(train_features_json, test_features_json, args.knn_factor, n_windows)

def classify_images(train_features_json, test_features_json, knn_factor, n_windows):
    train_data, train_labels = get_features(train_features_json)
    test_data, test_labels = get_features(test_features_json)

    for kernel in train_data.keys():
        if kernel == "label":
            continue

        train_data[kernel] = np.array(train_data[kernel])
        test_data[kernel] = np.array(test_data[kernel])

        knn = KNeighborsClassifier(n_neighbors=int(knn_factor))
        knn.fit(train_data[kernel], train_labels[kernel])

        predictions = knn.predict(test_data[kernel])
        accuracy = accuracy_score(test_labels[kernel], predictions)

        print(f"Kernel: {kernel}, Accuracy: {accuracy:.2f}")

def get_features(features_json):
    data = {}
    labels = {}

    for image_path in features_json.keys():
        for kernel in features_json[image_path]:
            if kernel == "label":
                continue

            if kernel not in data:
                data[kernel] = []
                labels[kernel] = []

            filter_names = list(features_json[image_path][kernel].keys())
            n_windows = len(features_json[image_path][kernel][filter_names[0]]["features"])
            
            windows = [[] for _ in range(n_windows)]

            for filter_name in filter_names:
                filter_features = features_json[image_path][kernel][filter_name]["features"]
                for i, feat in enumerate(filter_features):
                    windows[i].append(feat)

            flat_features = [np.mean(f) for f in windows]
            data[kernel].append(flat_features)
            labels[kernel].append(features_json[image_path]["label"])

    return data, labels

def get_n_windows(features_dir):
    info_json = json.load(open(features_dir + "args.json", "r"))

    return (info_json["resize_size"] // info_json["window_size"])**2

def handle_args(args):
    if not os.path.exists(args.features_dir):
        raise FileNotFoundError(f"Features directory {args.features_dir} does not exist.")
    
    args.features_dir += "/" if not args.features_dir.endswith("/") else ""

def get_args():
    parser = argparse.ArgumentParser(description="Segmentation of images")

    parser.add_argument("--features_dir", type=str, required=True, help="Path to the output of extract_textures.py")
    parser.add_argument("--knn_factor", type=str, required=True, help="KNN number of neighbors")

    return parser.parse_args()

if __name__ == "__main__":
    main()
