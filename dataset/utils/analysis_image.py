import os
import json
import re
import cv2
from tqdm import tqdm


def analyze_image(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 获取图像大小
    image_height, image_width = image.shape
    image_area = image_height * image_width

    # 检测白色物体
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化结果列表
    results = {
        "object_count": 0,
        "objects": [],
        "total_area_ratio": 0,
        "image_size": (image_width, image_height)
    }

    # 计算每个轮廓的面积和边界框
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        if area_ratio > 0.001:
            results["objects"].append({
            "bbox": [round(x / image_width, 6), round(y / image_height, 6),
                    round((x + w) / image_width, 6), round((y + h) / image_height, 6)],
            "area": area,
            "area_ratio": area_ratio
            })
        results["total_area_ratio"] += area_ratio
    results["object_count"] = len(results['objects'])
    return results