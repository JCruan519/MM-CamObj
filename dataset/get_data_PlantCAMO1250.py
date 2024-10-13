import os
import json
import re
import cv2
from tqdm import tqdm
from utils.analysis_image import analyze_image



def get_name(str):
    name_list = str.split('_')
    if name_list[-1].isdigit():
        name_list = name_list[-3:-1]
        return ' '.join(name_list)
    else:
        name_list = name_list[-3:-1]
        return ' '.join(name_list)


# data_path = 'data/PlantCAMO1250'


def process_PlantCAMO1250(data_path):
    # data_path = 'data/PlantCAMO1250'
    total_images = []
    test_gt = [os.path.join(data_path, "test", "gt", gt) for gt in os.listdir(os.path.join(data_path, "test", "gt"))]
    test_rgb = [os.path.join(data_path, "test", "rgb", rgb) for rgb in
                os.listdir(os.path.join(data_path, "test", "rgb"))]
    train_gt = [os.path.join(data_path, "train", "gt", gt) for gt in os.listdir(os.path.join(data_path, "train", "gt"))]
    train_rgb = [os.path.join(data_path, "train", "rgb", rgb) for rgb in
                 os.listdir(os.path.join(data_path, "train", "rgb"))]

    for i in tqdm(range(len(test_gt)),desc='PlantCAMO1250_test'):
        img_dict = {
            "dataset" :"PlantCAMO1250",
            "unique_id": "",
            "base_class": get_name(test_gt[i].split('/')[-1]),
            "image": test_rgb[i],
            "mask": test_gt[i],
            "id": 0,
            "bbox": [objects['bbox'] for objects in analyze_image(test_gt[i])['objects']],
            "analysis_img": analyze_image(test_gt[i])
        }
        total_images.append(img_dict)

    for i in tqdm(range(len(train_gt)),desc='PlantCAMO1250_train'):
        img_dict = {
            "dataset" :"PlantCAMO1250",
            "unique_id": "",
            "base_class": get_name(train_gt[i].split('/')[-1]),
            "image": train_rgb[i],
            "mask": train_gt[i],
            "id": 0,
            "bbox": [objects['bbox'] for objects in analyze_image(train_gt[i])['objects']],
            "analysis_img": analyze_image(train_gt[i])
        }
        total_images.append(img_dict)
    print(len(total_images))
    return total_images


# data_dict = process_PlantCAMO1250(data_path)

# print(data_dict)
# print(len(data_dict))
