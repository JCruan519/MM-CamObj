"""pip install accelerate transformers>=4.35.2
BLIP_FLANT5 tends to otuput shorter text, like "a tiger and a zebra". Try to design the prompt with shorter answer.
"""
import requests
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from typing import List
import torch
from typing import List
from io import BytesIO
from utils import merge_images, load_image
import traceback

class INSTRUCTBLIP_FLANT5():
    support_multi_image = False
    def __init__(self, model_path:str="Salesforce/instructblip-flan-t5-xxl", eval_mode=None) -> None:
        """
        Args:
            model_path (str): BLIP_FLANT5 model name, e.g. "Salesforce/blip2-flan-t5-xxl"
        """
        self.model_path = model_path
        self.processor = InstructBlipProcessor.from_pretrained(model_path)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        self.eval_mode = eval_mode
    
    def __call__(self, inputs: dict) -> str:
        if self.eval_mode.startswith('easy_VQA'):
            try:
                generated_text = self.get_VQA_answer(inputs)
            except:
                traceback.print_exc()
                return 'ERROR!!!'
        elif self.eval_mode.startswith('hard_VQA'):
            try:
                generated_text = self.get_VQA_answer(inputs)
            except:
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
                traceback.print_exc()
                return 'ERROR!!!'
        elif self.eval_mode.startswith('mask_match'):
            try:
                generated_text = self.get_mask_match_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('mask_FT'):
            try:
                generated_text = self.get_mask_FT_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('VQA') or self.eval_mode.startswith('image_cap'):
            try:
                generated_text = self.get_general_answer(inputs)
            except:
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
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            return self.get_parsed_output(inputs)
        

    def get_general_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            return self.get_parsed_output(inputs)
        
    def get_bbox_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        additional_prompt = "Please provide your boundingbox answer in the format [x1, y1, x2, y2], where (x1, y1) are the coordinates of the top-left corner of the bbox, and (x2, y2) are the coordinates of the bottom-right corner, where x and y are float numbers in the range of [0,1]."
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = inputs['question'] + additional_prompt
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            return self.get_parsed_output(inputs)
        
    def get_count_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            return self.get_parsed_output(inputs)
        
    def get_mask_match_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image_list': 
            }
        """
        additional_prompt = 'Please note that your answer must be "A" or "B".'
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = inputs['question']+additional_prompt
            inputs = self.prepare_prompt(inputs['image_list'], text_prompt+"")
            return self.get_parsed_output(inputs)
    def get_mask_FT_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image_list': 
            }
        """
        additional_prompt = 'Please note that your answer must be "True" or "False".'
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = inputs['question']+additional_prompt
            inputs = self.prepare_prompt(inputs['image_list'], text_prompt+"")
            return self.get_parsed_output(inputs)
        
        
    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        image = merge_images(image_links)
        inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def get_parsed_output(self, inputs):
        generation_output = self.model.generate(**inputs, 
            do_sample=True,
            # num_beams=5,
            max_new_tokens=512,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        return generation_text[0].strip(" \n")
 