"""pip install timm sentencepiece
"""
import os
import torch
import time
import torch.nn as nn
from typing import List, Union, Optional, Dict
from transformers.image_utils import load_image
import os
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import requests
import random
from tqdm import tqdm
import json
import io





class GPT4V():
    def __init__(self, model_path:str="gpt-4o-mini", eval_mode=None) -> None:
        self.model_path = model_path
        self.eval_mode = eval_mode

        print(model_path)
    
    def __call__(self, inputs: dict) -> str:
        if self.eval_mode.startswith('single_choice'):
            try:
                generated_text = self.get_single_choice_anwser(inputs)
            except:
                try: generated_text = self.get_single_choice_anwser(inputs)
                except: 
                    try: generated_text = self.get_single_choice_anwser(inputs)
                    except:
                        try: generated_text = self.get_single_choice_anwser(inputs)
                        except:
                            return 'ERROR!!!'
        elif self.eval_mode.startswith('flow_insert'):
            try:
                generated_text = self.get_flow_insert_answer(inputs)
            except:
                try: generated_text = self.get_flow_insert_answer(inputs)
                except: 
                    try: generated_text = self.get_flow_insert_answer(inputs)
                    except:
                        try: generated_text = self.get_flow_insert_answer(inputs)
                        except:
                            return 'ERROR!!!'
        elif self.eval_mode.startswith('fake_news'):
            try:
                generated_text = self.get_fake_news_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('image_cap'):
            try:
                generated_text = self.get_image_cap_answer(inputs)
            except:
                try: generated_text = self.get_image_cap_answer(inputs)
                except: 
                    try: generated_text = self.get_image_cap_answer(inputs)
                    except:
                        try: generated_text = self.get_image_cap_answer(inputs)
                        except:
                            return 'ERROR!!!'
        else:
            raise NotImplementedError
        return generated_text


    def get_single_choice_anwser(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'above_content':
                'below_content: 
                'images': [
                    
                ]
                'temple_img': 
                'temple_txt': 
            }
        """
        temple_txt = inputs['temple_txt']
        temple_img = inputs['temple_img']

        user_prompt = temple_txt + inputs['above_content'] + '\n' + inputs['below_content'] + '\n' + temple_img
        return get_res_by_gpt(user_prompt, inputs['images'], model_name=self.model_path)


    def get_flow_insert_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'paragraphs': 
                'image': 
                'temple_img': 
                'temple_txt': 
            }
        """
        temple_txt = inputs['temple_txt']
        temple_img = inputs['temple_img']

        user_prompt = temple_txt + inputs['paragraphs'] + '\n' + temple_img
        return get_res_by_gpt(user_prompt, inputs['image'], model_name=self.model_path)
    
    def get_image_cap_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question':
                'image':
            }
        """

        user_prompt = inputs['question']
        return get_res_by_gpt(user_prompt, inputs['image'], model_name=self.model_path)

def compress_image(image_path, max_size_mb=10):
    max_size_bytes = max_size_mb * 1024 * 1024
    img = Image.open(image_path)
    
    # 压缩图片直到大小小于max_size_bytes
    quality = 95
    while True:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        size = buffer.tell()
        if size <= max_size_bytes or quality <= 10:
            break
        quality -= 5
    
    buffer.seek(0)
    return buffer

def encode_image_to_base64(image_path):
    image_size = os.path.getsize(image_path)
    max_size_bytes = 10 * 1024 * 1024
    
    if image_size > max_size_bytes:
        compressed_image = compress_image(image_path)
        encoded_image = base64.b64encode(compressed_image.read()).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    return encoded_image

def get_res_by_gpt(user_prompt, image_path_list, model_name):

    url = "http://47.88.65.188:8300/v1/chat/completions"

    content = []
    content.append(
        {
            "type": "text",
            "text": user_prompt
        }
    )
    if type(image_path_list) == str:
        encoded_image = encode_image_to_base64(image_path_list)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpg;base64,{encoded_image}"
                }
            }
        )
    else:
        for img_path in image_path_list:
            encoded_image = encode_image_to_base64(img_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{encoded_image}"
                    }
                }
            )

    # 构造请求的载荷
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        # "max_tokens": 500,  # 设置生成文本的最大长度
        # "temperature": 1.0,  # 控制生成文本的随机性
        # "top_p": 1.0,  # 控制生成文本的多样性
        # "n": 1,  # 生成一个响应
        # "stop": ["\n"]  # 指定停止生成的标志
    })
    
    # 请求头
    headers = {
        'Authorization': 'sk-HULl5F78opXSt7f65dF86362A7Fd496c92DeBa06692a8a14',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': 'www.apihy.com',
        'Connection': 'keep-alive'
    }
    
    # 发送POST请求
    response = requests.post(url, headers=headers, data=payload)
    
    # 返回响应文本
    return json.loads(response.text, strict=False)["choices"][0]['message']['content']