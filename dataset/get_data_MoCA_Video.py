import os
import json
import re
import cv2
from tqdm import tqdm
from utils.analysis_image import analyze_image


def get_name(str):
    name_list = str.split('_')
    if name_list[-1].isdigit():
        name_list = name_list[:-1]
        return ' '.join(name_list)
    else:
        return ' '.join(name_list)


# data_path = 'data/MoCA_Video'


def process_MoCA_Video(data_path):
    # data_path = 'data/MoCA_Video'
    total_images = []
    TestDataset_per_sq = os.path.join(data_path, "TestDataset_per_sq")
    TrainDataset_per_sq = os.path.join(data_path, "TrainDataset_per_sq")
    Test_subdir = [os.path.join(TestDataset_per_sq, subdir) for subdir in os.listdir(TestDataset_per_sq)]
    Train_subdir = [os.path.join(TrainDataset_per_sq, subdir) for subdir in os.listdir(TrainDataset_per_sq)]
    for subdir in tqdm(Test_subdir,desc='MoCA_Vedio_Test'):
        subdir_path_GTs = [os.path.join(subdir, 'GT', GT) for GT in os.listdir(os.path.join(subdir, 'GT'))]
        subdir_path_Imgs = [os.path.join(subdir, 'Imgs', Imgs) for Imgs in os.listdir(os.path.join(subdir, 'Imgs'))]
        for i in range(len(subdir_path_GTs)):
            # print("/".join(subdir_path_GTs[i].split('/')[:-1]+[subdir_path_Imgs[i].split('/')[-1].split('.')[0]+'.png']))
            img_dict = {
                "dataset" :"MoCA_Video",
                "unique_id": "",
                "base_class": get_name(subdir_path_GTs[i].split('/')[-3]),
                "image": subdir_path_Imgs[i],
                "mask": "/".join(subdir_path_GTs[i].split('/')[:-1]+[subdir_path_Imgs[i].split('/')[-1].split('.')[0]+'.png']),
                "id": 0,
                "bbox": [objects['bbox'] for objects in analyze_image("/".join(subdir_path_GTs[i].split('/')[:-1]+[subdir_path_Imgs[i].split('/')[-1].split('.')[0]+'.png']))['objects']],
                "analysis_img": analyze_image("/".join(subdir_path_GTs[i].split('/')[:-1]+[subdir_path_Imgs[i].split('/')[-1].split('.')[0]+'.png']))
            }
            total_images.append(img_dict)

    for subdir in tqdm(Train_subdir,desc='MoCA_Vedio_Train'):
        subdir_path_GTs = [os.path.join(subdir, 'GT', GT) for GT in os.listdir(os.path.join(subdir, 'GT'))]
        subdir_path_Imgs = [os.path.join(subdir, 'Imgs', Imgs) for Imgs in os.listdir(os.path.join(subdir, 'Imgs'))]
        for i in range(len(subdir_path_GTs)):
            img_dict = {
                "dataset" :"MoCA_Video",
                "unique_id": "",
                "base_class": get_name(subdir_path_GTs[i].split('/')[-3]),
                "image": subdir_path_Imgs[i],
                "mask": "/".join(subdir_path_GTs[i].split('/')[:-1]+[subdir_path_Imgs[i].split('/')[-1].split('.')[0]+'.png']),
                "id": 0,
                "bbox": [objects['bbox'] for objects in analyze_image("/".join(subdir_path_GTs[i].split('/')[:-1]+[subdir_path_Imgs[i].split('/')[-1].split('.')[0]+'.png']))['objects']],
                "analysis_img": analyze_image("/".join(subdir_path_GTs[i].split('/')[:-1]+[subdir_path_Imgs[i].split('/')[-1].split('.')[0]+'.png']))
            }
            total_images.append(img_dict)
    print(len(total_images))
    return total_images

# data_path = "data"+"/MoCA_Video"
# data_dict = process_MoCA_Video(data_path)

# print(data_dict)
# print(len(data_dict))
