import os
import sys
import json
from glob import glob
import numpy as np
import cv2

class_to_id = {
    "road": 0,
    "motorcycle": 1,
    "bridge": 2,
    "caravan": 3,
    "motorcyclegroup": 4,
    "guard rail": 5,
    "sky": 6,
    "wall": 7,
    "pole": 8,
    "bicycle": 9,
    "tunnel": 10,
    "traffic light": 11,
    "dynamic": 12,
    "fence": 13,
    "bus": 14,
    "static": 15,
    "out of roi": 16,
    "truckgroup": 17,
    "ground": 18,
    "trailer": 19,
    "ego vehicle": 20,
    "sidewalk": 21,
    "polegroup": 22,
    "rider": 23,
    "persongroup": 24,
    "building": 25,
    "cargroup": 26,
    "car": 27,
    "train": 28,
    "rectification border": 29,
    "bicyclegroup": 30,
    "ridergroup": 31,
    "person": 32,
    "terrain": 33,
    "parking": 34,
    "rail track": 35,
    "traffic sign": 36,
    "license plate": 37,
    "vegetation": 38,
    "truck": 39
}

def main(leftImg8bit, gtFine, output_folder):
    images = glob(f"{leftImg8bit}/**/*.png", recursive=True)

    for image in images:
        image_name = os.path.basename(image)
        image_city = image.split('/')[-2]
        image_mode = image.split('/')[-3]

        json_path = f"{gtFine}/{image_mode}/{image_city}/{image_name.replace('leftImg8bit.png', 'gtFine_polygons.json')}"
        json_file = json.load(open(json_path, 'r'))

        height = json_file['imgHeight']
        width = json_file['imgWidth']

        polygons = {}
        for object in json_file['objects']:
            label = object['label']
            polygon = object['polygon']

            if label not in polygons:
                polygons[label] = []
        
            polygons[label].append(polygon)

        # Creating masks
        for label, polygons_list in polygons.items():
            label_id = class_to_id[str(label)]

            # Skipping classes 16, 20, 29
            if label_id in [16, 20, 29]:
                continue

            mask = create_mask(polygons_list, height, width)
            mask_name = f"{str(label_id)}_{image_name}"

            output_mask_path = f"{output_folder}/{image_mode}/{image_city}/{mask_name}"
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

            save_mask(mask, output_mask_path)

def create_mask(polygons_list, height, width):
    im = np.zeros((height, width), dtype=np.uint8)

    for polygon in polygons_list:
        polygon = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(im, [polygon], 1)

    return im

def save_mask(mask, output_path):
    mask[mask > 0] = 255
    cv2.imwrite(output_path, mask)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_mask_folder.py <leftImg8bit> <gtFine> <output_folder>")
        sys.exit(1)

    leftImg8bit = sys.argv[1]
    gtFine = sys.argv[2]
    output_folder = sys.argv[3]

    leftImg8bit += "/" if not leftImg8bit.endswith('/') else ""
    gtFine += "/" if not gtFine.endswith('/') else ""
    output_folder += "/masks/" if not output_folder.endswith('/') else "masks/"

    os.makedirs(output_folder, exist_ok=True)

    main(leftImg8bit, gtFine, output_folder)
