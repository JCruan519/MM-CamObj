import os
import json
import re
import cv2
from tqdm import tqdm
from utils.analysis_image import analyze_image


def get_name(str):
    name_list = str.split('_')
    
    return name_list[0]


# data_path = 'data/CamouflageData'


def process_CamouflageData(data_path):
    dataset_dict = {
    "dataset01": "Arid Fleck",
    "dataset02": "British DPM",
    "dataset03": "CADPAT",
    "dataset04": "Danish M",
    "dataset05": "Desert Digital MARPAT",
    "dataset06": "Desert DPM",
    "dataset07": "Desert Tiger Stripe",
    "dataset08": "German Snow Camouflage",
    "dataset09": "MARPAT Digital Woodland",
    "dataset10": "Rhodesian Pattern",
    "dataset11": "BGS Sumpfmuster",
    "dataset12": "British Multi-Terrain Pattern",
    "dataset13": "Coyote Tan",
    "dataset14": "Czech VZ 95",
    "dataset15": "Desert Night",
    "dataset16": "French CCE",
    "dataset17": "German Flecktarn",
    "dataset18": "German WWII 44 Dot",
    "dataset19": "IDF Combat Uniform",
    "dataset20": "Kryptek Mandrake"
    }
    # data_path = 'data/CamouflageData'
    total_images = []
    gt = [os.path.join(data_path, "gt", gt) for gt in os.listdir(os.path.join(data_path, "gt"))]
    img = [os.path.join(data_path, "img", img) for img in os.listdir(os.path.join(data_path, "img"))]

    for i in tqdm(range(len(gt)) ,desc='CamouflageData'):
        img_dict = {
            "dataset" :"CamouflageData",
            "unique_id": "",
            "base_class": 'human_'+dataset_dict[get_name(gt[i].split('/')[-1])],
            "image": img[i],
            "mask": "/".join(gt[i].split('/')[:-1]+[img[i].split('/')[-1].split('.')[0]+'.png']),
            "id": 0,
            "bbox": [objects['bbox'] for objects in analyze_image(gt[i])['objects']],
            "analysis_img": analyze_image(gt[i])
        }
        total_images.append(img_dict)
    print(len(total_images))
    return total_images


# data_dict = process_CamouflageData(data_path)

# print(data_dict)
# print(len(data_dict))
