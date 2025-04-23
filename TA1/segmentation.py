import os
import json
import argparse
import numpy as np
from scipy.spatial.distance import euclidean
from PIL import Image

def main():
    args = get_args()
    handle_args(args)

    resize_size, window_size = get_shape_info(args.features_dir)

    train_features_json = json.load(open(args.features_dir + "train_features.json", "r"))
    test_features_json = json.load(open(args.features_dir + "test_features.json", "r"))

    segment_images(train_features_json, args.output, args.limiar, resize_size, window_size)
    segment_images(test_features_json, args.output, args.limiar, resize_size, window_size)

def segment_images(features_json, output_dir, limiar, resize_size, window_size):
    for image_path in features_json.keys():
        for kernel in features_json[str(image_path)].keys():
            if kernel == "label":
                continue
                
            features = []
            feito = False

            for filter_name in features_json[image_path][kernel].keys():
                filter_features = features_json[image_path][kernel][filter_name]["features"]

                for i, feature in enumerate(filter_features):
                    if not feito:
                        features.append([])
                    
                    features[i].append(feature)
                
                feito = True
            
            region_classes = {i: -1 for i in range(len(features))}

            for region_idx in range(len(features)):
                region = features[region_idx]

                for other_region_idx in range(len(features)):
                    if region_idx == other_region_idx or region_classes[other_region_idx] != -1:
                        continue

                    other_region = features[other_region_idx]
                    distance = euclidean(region, other_region)

                    if distance < limiar:
                        if region_classes[region_idx] == -1:
                            max_i = max(region_classes.values())
                            region_classes[region_idx] = max_i + 1
                        
                        region_classes[other_region_idx] = region_classes[region_idx]
            
            max_i = max(region_classes.values())
            for region_class in region_classes:
                if region_classes[region_class] == -1:
                    max_i += 1
                    region_classes[region_class] = max_i
            
            save_image(output_dir, image_path, region_classes, resize_size, window_size, kernel)

def save_image(output_dir, image_path, region_classes, resize_size, window_size, kernel):
    image_name = os.path.basename(image_path)
    image_class = image_path.split("/")[-2]

    class_dir = os.path.join(output_dir, image_class)
    os.makedirs(class_dir, exist_ok=True)

    segmented_image = np.zeros((resize_size, resize_size), dtype=np.uint8)

    idx = 0
    for i in range(0, resize_size, window_size):
        for j in range(0, resize_size, window_size):
            region_class = region_classes[idx]
            segmented_image[i:i+window_size, j:j+window_size] = region_class
            idx += 1

    
    min_val = np.min(segmented_image)
    max_val = np.max(segmented_image)

    if max_val > min_val:
        segmented_image = ((segmented_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        segmented_image = np.zeros_like(segmented_image, dtype=np.uint8)


    original_image = Image.open(image_path).resize((resize_size, resize_size))
    segmented_pil = Image.fromarray(segmented_image)

    combined_width = resize_size * 2
    combined_image = Image.new("RGB", (combined_width, resize_size))
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(segmented_pil, (resize_size, 0))

    combined_image.save(f"{class_dir}/{str(kernel)}_{image_name}", "JPEG")

def get_shape_info(features_dir):
    info_json = json.load(open(features_dir + "args.json", "r"))

    return info_json["resize_size"], info_json["window_size"]

def handle_args(args):
    if not os.path.exists(args.features_dir):
        raise FileNotFoundError(f"Features directory {args.features_dir} does not exist.")
    
    os.makedirs(args.output, exist_ok=True)

    args.output += "/" if not args.output.endswith("/") else ""
    args.features_dir += "/" if not args.features_dir.endswith("/") else ""

def get_args():
    parser = argparse.ArgumentParser(description="Segmentation of images")

    parser.add_argument("--features_dir", type=str, required=True, help="Path to the output of extract_textures.py")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--limiar", type=float, required=True, help="Distance limiar for segmentation")

    return parser.parse_args()

if __name__ == "__main__":
    main()
