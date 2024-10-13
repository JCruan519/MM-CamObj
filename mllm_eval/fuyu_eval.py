"""need latest transformers from source
pip install transformers>=4.35.2
"""
import requests
import torch
from transformers import FuyuProcessor, FuyuForCausalLM, AutoTokenizer
from PIL import Image
from typing import List
from io import BytesIO
from utils import merge_images, load_image
import re
import traceback

class Fuyu():
    support_multi_image = False
    def __init__(self, model_path:str="adept/fuyu-8b", eval_mode=None) -> None:
        """
        Args:
            model_path (str): Fuyu model name, e.g. "adept/fuyu-8b"
        """
        self.model_path = model_path
        self.processor = FuyuProcessor.from_pretrained(model_path)
        self.model = FuyuForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.eval_mode = eval_mode
    
    def __call__(self, inputs: dict) -> str:
        if  self.eval_mode.startswith('easy_VQA'):
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
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = temple_txt + inputs['above_content'] + '\n' + inputs['below_content'] + '\n' + temple_img
            inputs = self.prepare_prompt(inputs['images'], text_prompt)
            return self.get_parsed_output(inputs)


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
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = temple_txt + inputs['paragraphs'] + '\n' + temple_img
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            return self.get_parsed_output(inputs)

          
    def extract_answers(self,text):
        """
        Extracts multiple-choice answers from the given text, 
        and trims whitespace or other extraneous characters from the ends of each answer.
        
        Args:
        text (str): The text containing the multiple-choice answers.
        
        Returns:
        list: A list of trimmed answers.
        """
        # Regular expression to find the answers
        pattern = r"[A-D]\.\s+(.*?)\n"
        matches = re.findall(pattern, text)
        # Trim whitespace or other extraneous characters from each answer
        trimmed_matches = [match.strip() for match in matches]
        return trimmed_matches

    def get_VQA_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        answer_list = self.extract_answers(inputs['question'])
        if self.support_multi_image:    
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            model_answer = self.get_parsed_output(inputs)
            no_spaces_model_answer = model_answer.replace(" ", "")
            print(model_answer)
            # 直接检查 model_answer 是否是 'A', 'B', 'C' 或 'D'
            if no_spaces_model_answer in ['A', 'B', 'C', 'D']:
                model_answer = no_spaces_model_answer
            # 否则，检查 answer_list 中去除空格后的选项是否在 model_answer 中
            elif answer_list[0].replace(" ", "") in no_spaces_model_answer:
                model_answer = 'A'
            elif answer_list[1].replace(" ", "") in no_spaces_model_answer:
                model_answer = 'B'
            elif answer_list[2].replace(" ", "") in no_spaces_model_answer:
                model_answer = 'C'
            elif answer_list[3].replace(" ", "") in no_spaces_model_answer:
                model_answer = 'D'
            else:
                model_answer = no_spaces_model_answer  # 如果没有匹配到任何选项，设置为 'Unknown'

            return model_answer
        
    def get_bbox_answer(self, inputs: dict) -> str:
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
        

    def get_mask_match_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image_list': 
            }
        """
        if self.support_multi_image:    
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            inputs = self.prepare_prompt(inputs['image_list'], text_prompt)
            return self.get_parsed_output(inputs)
    def get_mask_FT_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image_list': 
            }
        """
        if self.support_multi_image:    
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            inputs = self.prepare_prompt(inputs['image_list'], text_prompt)
            return self.get_parsed_output(inputs)
    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        image = merge_images(image_links)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(self.model.device)
        return inputs
    
    def get_parsed_output(self, inputs):
        generation_output = self.model.generate(**inputs, max_new_tokens=512, pad_token_id=self.pad_token_id)
        input_len = inputs.input_ids.shape[1]
        generation_text = self.processor.batch_decode(generation_output[:, input_len:], skip_special_tokens=True)
        return generation_text[0].strip(" \n")
