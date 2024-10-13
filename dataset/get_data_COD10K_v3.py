import os
import json
import re
import cv2
from tqdm import tqdm
from utils.analysis_image import analyze_image

# data_path = 'data/COD10K-v3'
pattern = re.compile(r'^COD10K-CAM-(\d+)-([A-Za-z]+)-(\d+)-([A-Za-z]+)-(\d+)\.jpg$')


def process_COD10K_v3(data_path):
    # data_path = 'data/COD10K-v3'
    total_images = []
    Test_GT = [os.path.join(data_path, 'Test', 'GT_Object', GT) for GT in
               os.listdir(os.path.join(data_path, 'Test', 'GT_Object'))]
    Test_image = [os.path.join(data_path, 'Test', 'Image', image) for image in
                  os.listdir(os.path.join(data_path, 'Test', 'Image'))]
    Train_GT = [os.path.join(data_path, 'Train', 'GT_Object', GT) for GT in
                os.listdir(os.path.join(data_path, 'Train', 'GT_Object'))]
    Train_image = [os.path.join(data_path, 'Train', 'Image', image) for image in
                   os.listdir(os.path.join(data_path, 'Train', 'Image'))]
    for i in tqdm(range(len(Test_image)),desc='COD10K_v3_TEST'):
        match = pattern.match(Test_image[i].split('/')[-1])
        if match:
            super_number, super_class, sub_number, subclass, image_number = match.groups()
            if subclass == 'Other':
                continue
            img_dict = {
                "dataset" :"COD10K_v3",
                "unique_id": "",
                "base_class": subclass,
                "image": Test_image[i],
                "mask": Test_GT[i],
                "id": 0,
                "bbox": [objects['bbox'] for objects in analyze_image(Test_GT[i])['objects']],
                "analysis_img": analyze_image(Test_GT[i])
            }
            total_images.append(img_dict)

            # if i >10:
            #     break
    for i in tqdm(range(len(Train_image)),desc='COD10K_v3_TRAIN'):

        match = pattern.match(Train_image[i].split('/')[-1])
        if match:
            super_number, super_class, sub_number, subclass, image_number = match.groups()
            if subclass == 'Other':
                continue
            img_dict = {
                "dataset" :"COD10K_v3",
                "unique_id": "",
                "base_class": subclass,
                "image": Train_image[i],
                "mask": Train_GT[i],
                "id": 0,
                "bbox": [objects['bbox'] for objects in analyze_image(Train_GT[i])['objects']],
                "analysis_img": analyze_image(Train_GT[i])
            }
            total_images.append(img_dict)
    print(len(total_images))
    return total_images


# data_dict = process_COD10K_v3(data_path)
# for line in data_dict:
#     print(line)
# print(len(data_dict))
