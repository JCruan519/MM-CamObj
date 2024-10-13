"""pip install transformers>=4.35.2
"""
import os
import torch
import time
from typing import List
from transformers import IdeficsForVisionText2Text, AutoProcessor
import re
import traceback

class Idefics1():
    support_multi_image = True
    merged_image_files = []
    def __init__(self, model_path:str="HuggingFaceM4/idefics-9b-instruct", eval_mode=None) -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        self.model = IdeficsForVisionText2Text.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.eval_mode = eval_mode

        
    def __call__(self, inputs: dict) -> str:
        if self.eval_mode.startswith('easy_VQA'):
            try:
                generated_text = self.get_VQA_answer(inputs)
            except:
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
        elif self.eval_mode.startswith('count_choice'):
            try:
                generated_text = self.get_count_choice_answer(inputs)
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
            prompt = ["USER: "]
            prompt += [inputs['question'] , '\n' , inputs['image']]
            prompt += ["<end_of_utterance>", "\nAssistant:"]
            inputs = self.processor([prompt], return_tensors="pt").to(self.model.device)
            bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
            generated_ids = self.model.generate(**inputs, bad_words_ids=bad_words_ids, max_new_tokens=256)
            generated_text = self.processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # 正则表达式匹配第一个大写字母后跟句号的模式
            pattern = r"\b[A-Z]\."

            # 使用search找到第一个匹配的项
            match = re.search(pattern, generated_text)

            # 如果找到匹配项，打印出来
            if match:
                first_answer = match.group(0)
            else:
                first_answer = 'A'

            return first_answer
        else:
            raise NotImplementedError
        
    def get_bbox_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        if self.support_multi_image:
            additional_prompt = "\nDo not include any code in your answer. If you cannot provide the bbox, first provide the target's coordinates and size, then convert these to a bounding box."
            prompt = ["USER: "]
            prompt += [inputs['question'] ,additional_prompt, '\n' , inputs['image']]
            prompt += ["<end_of_utterance>", "\nAssistant:"]
            inputs = self.processor([prompt], return_tensors="pt").to(self.model.device)
            bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
            generated_ids = self.model.generate(**inputs, bad_words_ids=bad_words_ids, max_new_tokens=256)
            generated_text = self.processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            return generated_text
        else:
            raise NotImplementedError

    def get_count_choice_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        additional_pprompt = "\nPlease note that your answer must be in A,B,C,D"
        if self.support_multi_image:
            prompt = ["USER: "]
            prompt += [inputs['question']+additional_pprompt , '\n' , inputs['image']]
            prompt += ["<end_of_utterance>", "\nAssistant:"]
            inputs = self.processor([prompt], return_tensors="pt").to(self.model.device)
            bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
            generated_ids = self.model.generate(**inputs, bad_words_ids=bad_words_ids, max_new_tokens=256)
            generated_text = self.processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # 正则表达式匹配第一个大写字母后跟句号的模式
            pattern = r"\b[A-Z]\."

            # 使用search找到第一个匹配的项
            match = re.search(pattern, generated_text)

            # 如果找到匹配项，打印出来
            if match:
                first_answer = match.group(0)
            else:
                first_answer = 'A'

            return first_answer
        else:
            raise NotImplementedError
            
    def get_general_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            } """  
        if self.support_multi_image:
            prompt = ["USER: ", inputs['image'], inputs['question'] , "<end_of_utterance>", "\nAssistant:"]
            inputs = self.processor([prompt], return_tensors="pt").to(self.model.device)
            bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
            generated_ids = self.model.generate(**inputs, bad_words_ids=bad_words_ids, max_new_tokens=2048)
            generated_text = self.processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            return generated_text
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
            prompt = ["USER: "]
            prompt += [inputs['question'] , '\n']
            for img in inputs['image_list']:
                prompt.append(img)
            prompt += ["<end_of_utterance>", "\nAssistant:"]
            inputs = self.processor([prompt], return_tensors="pt").to(self.model.device)
            bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
            generated_ids = self.model.generate(**inputs, bad_words_ids=bad_words_ids, max_new_tokens=256)
            generated_text = self.processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if "'A'" in generated_text or " A " in generated_text or 'first' in generated_text:
                generated_text = "A"
            elif "'B'" in generated_text or " B " in generated_text or 'second' in generated_text:
                generated_text = "B"

            return generated_text
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
            prompt = ["USER: "]
            prompt += [inputs['question'] , '\n']
            for img in inputs['image_list']:
                prompt.append(img)
            prompt += ["<end_of_utterance>", "\nAssistant:"]
            inputs = self.processor([prompt], return_tensors="pt").to(self.model.device)
            bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
            generated_ids = self.model.generate(**inputs, bad_words_ids=bad_words_ids, max_new_tokens=256)
            generated_text = self.processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            return generated_text
        else:
            raise NotImplementedError
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
    