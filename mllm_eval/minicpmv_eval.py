"""pip install transformers>=4.35.2
"""
import os
import torch
import time
from PIL import Image
from typing import List
from transformers import AutoModel, AutoTokenizer
from transformers.image_utils import load_image
from transformers.utils import is_flash_attn_2_available
import re
import traceback

class MiniCPMV():
    support_multi_image = True
    def __init__(self, model_path:str="openbmb/MiniCPM-Llama3-V-2_5", eval_mode=None) -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda', _attn_implementation=attn_implementation).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.eval_mode = eval_mode

        print(f"Using {attn_implementation} for attention implementation")

        
    def __call__(self, inputs: dict) -> str:
        if self.eval_mode.startswith('easy_VQA'):
            try:
                generated_text = self.get_VQA_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('VQA') or self.eval_mode.startswith('image_cap'):
            try:
                generated_text = self.get_general_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('hard_VQA'):
            try:
                generated_text = self.get_VQA_answer(inputs)
            except:
                traceback.print_exc()
                return 'ERROR!!!'
        elif self.eval_mode.startswith('bbox'):
            try:
                generated_text = self.get_bbox_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('count'):
            try:
                generated_text = self.get_count_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('mask_match'):
            try:
                generated_text = self.get_mask_match_answer(inputs)
            except:
                traceback.print_exc()
                return 'ERROR!!!'
        elif self.eval_mode.startswith('mask_FT'):
            try:
                generated_text = self.get_mask_FT_answer(inputs)
            except:
                traceback.print_exc()
                return 'ERROR!!!'
        else:
            raise NotImplementedError
        return generated_text


            
    def get_VQA_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """  
        if self.support_multi_image:
            content = []
            text_prompt = inputs['question']
            content.append(text_prompt)
            if isinstance(inputs["image"], str): img = load_image(inputs["image"])
            elif isinstance(inputs["image"], Image.Image): img = inputs["image"]
            else: raise ValueError("Invalid image input", inputs["image"], "should be str or PIL.Image.Image")
            content.append(img)
         
            messages = [{"role": "user", "content": content}]
            res = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=False, # if sampling=False, beam_search will be used by default
            )
            return res
        else:
            raise NotImplementedError
        
    def get_general_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """  
        if self.support_multi_image:
            content = []
            text_prompt = inputs['question']
            content.append(text_prompt)
            if isinstance(inputs["image"], str): img = load_image(inputs["image"])
            elif isinstance(inputs["image"], Image.Image): img = inputs["image"]
            else: raise ValueError("Invalid image input", inputs["image"], "should be str or PIL.Image.Image")
            content.append(img)
         
            messages = [{"role": "user", "content": content}]
            res = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=False, # if sampling=False, beam_search will be used by default
            )
            return res
        else:
            raise NotImplementedError
      
    def extract_and_normalize_bbox(self,model_output, normalization_factor=1000):
    # 正则表达式匹配box标签内的坐标
        bbox_pattern = re.compile(r'<box>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)</box>')
        
        # 搜索匹配项
        match = bbox_pattern.search(model_output)
        if match:
            # 提取坐标
            x1, y1, x2, y2 = map(int, match.groups())
            
            # 归一化坐标，除以normalization_factor
            normalized_bbox = [
                x1 / normalization_factor,
                y1 / normalization_factor,
                x2 / normalization_factor,
                y2 / normalization_factor
            ]
            
            return normalized_bbox
        else:
            # 如果没有找到匹配项，返回None或抛出异常
            return None  
    def get_bbox_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """  
        if self.support_multi_image:
            prompt = 'please give your answer in the formart: <box>x1 y1 x2 y2</box>'
            content = []
            text_prompt = inputs['question'] + prompt
            content.append(text_prompt)
            if isinstance(inputs["image"], str): img = load_image(inputs["image"])
            elif isinstance(inputs["image"], Image.Image): img = inputs["image"]
            else: raise ValueError("Invalid image input", inputs["image"], "should be str or PIL.Image.Image")
            content.append(img)
         
            messages = [{"role": "user", "content": content}]
            res = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=False, # if sampling=False, beam_search will be used by default
            )
            return  str(self.extract_and_normalize_bbox(res))
        else:
            raise NotImplementedError
        
    def get_count_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """  
        if self.support_multi_image:
            content = []
            text_prompt = inputs['question']
            content.append(text_prompt)
            if isinstance(inputs["image"], str): img = load_image(inputs["image"])
            elif isinstance(inputs["image"], Image.Image): img = inputs["image"]
            else: raise ValueError("Invalid image input", inputs["image"], "should be str or PIL.Image.Image")
            content.append(img)
         
            messages = [{"role": "user", "content": content}]
            res = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=False, # if sampling=False, beam_search will be used by default
            )
            print(res)
            return res
        else:
            raise NotImplementedError
        
    def get_mask_match_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image_list': 
            }
        """  
        if self.support_multi_image:
            content = []
            text_prompt = "\nHere are three images. The first image is a picture containing a camouflaged creature, and the following two images are mask images that outline the camouflaged creature labeled as ['A', 'B']. Please select the mask image that best matches the camouflaged creature in the first image.  You need to respond with the label of the image that is the correct match and do not give any explanations\n"
            content.append(text_prompt)
            for image in inputs['image_list']:
                if isinstance(image, str): img = load_image(image)
                elif isinstance(image, Image.Image): img = image
                else: raise ValueError("Invalid image input", image, "should be str or PIL.Image.Image")
                content.append(img)
         
            messages = [{"role": "user", "content": content}]
            res = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=False, # if sampling=False, beam_search will be used by default
            )
            return res
        else:
            raise NotImplementedError
        
    def get_mask_FT_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image_list': 
            }
        """  
        if self.support_multi_image:
            content = []
            content.append(inputs["question"])
            for image in inputs['image_list']:
                if isinstance(image, str): img = load_image(image)
                elif isinstance(image, Image.Image): img = image
                else: raise ValueError("Invalid image input", image, "should be str or PIL.Image.Image")
                content.append(img)
         
            messages = [{"role": "user", "content": content}]
            res = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=False, # if sampling=False, beam_search will be used by default
            )
            return res
        else:
            raise NotImplementedError