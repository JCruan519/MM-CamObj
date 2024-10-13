"""pip install transformers>=4.35.2
"""
import os
import torch
import time
from typing import List
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers.utils import is_flash_attn_2_available
import traceback


class Idefics2():
    support_multi_image = True
    merged_image_files = []
    def __init__(self, model_path:str="HuggingFaceM4/idefics2-8b", eval_mode=None) -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, _attn_implementation=attn_implementation).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.eval_mode = eval_mode

        
    def __call__(self, inputs: dict) -> str:
        if self.eval_mode.startswith('single_choice'):
            try:
                generated_text = self.get_single_choice_anwser(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('flow_insert'):
            try:
                generated_text = self.get_flow_insert_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('fake_news'):
            try:
                generated_text = self.get_fake_news_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('easy_VQA'):
            try:
                generated_text = self.get_VQA_answer(inputs)
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
                traceback.print_exc()
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
        elif self.eval_mode.startswith('size_compare'):
            try:
                generated_text = self.get_size_compare_answer(inputs)
            except:
                traceback.print_exc()
                return 'ERROR!!!'
        elif self.eval_mode.startswith('VQA') or self.eval_mode.startswith('image_cap'):
            try:
                generated_text = self.get_general_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('location_choice'):
            try:
                generated_text = self.get_location_choice_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('mask_FT'):
            try:
                generated_text = self.get_mask_FT_answer(inputs)
            except:
                traceback.print_exc()
                return 'ERROR!!!'
        elif self.eval_mode.startswith('size_choice'):
            try:
                generated_text = self.get_size_choice_answer(inputs)
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

            text_prompt = temple_txt + inputs['above_content'] + '\n' + inputs['below_content'] + '\n' + temple_img
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * len(inputs['images']) + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(image_link) for image_link in inputs['images']]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        else:
            raise NotImplementedError
        

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
            text_prompt = temple_txt + inputs['paragraphs'] + '\n' + temple_img
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 1 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(inputs['image'])]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        else:
            raise NotImplementedError
            
            
    def get_VQA_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        if self.support_multi_image:
            text_prompt = inputs['question']
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 1 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(inputs['image'])]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        else:
            raise NotImplementedError

            
    def get_location_choice_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        if self.support_multi_image:
            text_prompt = inputs['question']
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 1 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(inputs['image'])]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        else:
            raise NotImplementedError

    def get_size_choice_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        if self.support_multi_image:
            text_prompt = inputs['question']
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 1 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(inputs['image'])]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
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
            text_prompt = inputs['question']
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 1 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(inputs['image'])]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=2048, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
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
        additional_prompt = "Please provide your bounding box (bbox) answer in the format [x1, y1, x2, y2], where (x1, y1) are the coordinates of the top-left corner of the bbox, and (x2, y2) are the coordinates of the bottom-right corner. All coordinate values should be normalized by dividing by the width or height of the image, ensuring they fall within the range of 0 to 1."
        if self.support_multi_image:
            text_prompt = inputs['question'] +additional_prompt
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 1 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(inputs['image'])]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if not generated_text.startswith('['):
                generated_text = generated_text.split(' ')
                generated_text = [float(x.strip('.')) for x in generated_text]
            return str(generated_text)
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
            text_prompt = inputs['question']
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 1 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(inputs['image'])]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
            text_prompt = inputs['question']
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 3 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(image) for image in inputs['image_list']]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
            text_prompt = inputs['question']
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] *2 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(image) for image in inputs['image_list']]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        else:
            raise NotImplementedError         
            
    def get_size_compare_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image_list': 
            }
        """
        if self.support_multi_image:
            text_prompt = inputs['question']
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 2 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(image) for image in inputs['image_list']]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if "A" in generated_text:
                generated_text = "A"
            elif "B" in generated_text:
                generated_text = "B"
            return generated_text
        else:
            raise NotImplementedError

        
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
