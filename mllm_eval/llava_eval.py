"""pip install transformers>=4.35.2
"""
import os
import tempfile
import requests
from PIL import Image
import torch
from io import BytesIO
from utils import merge_images, load_image
from conversation import conv_llava_v1
from typing import List
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils import is_flash_attn_2_available
import traceback

class Llava():
    support_multi_image = False
    merged_image_files = []
    def __init__(self, model_path:str="llava-hf/llava-1.5-7b-hf", eval_mode=None) -> None:
        """Llava model wrapper

        Args:
            model_path (str): Llava model name, e.g. "liuhaotian/llava-v1.5-7b" or "llava-hf/vip-llava-13b-hf"
        """
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation).eval()
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
        elif self.eval_mode.startswith('VQA') or self.eval_mode.startswith('image_cap'):
            try:
                generated_text = self.get_general_answer(inputs)
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
                traceback.print_exc()
                return 'ERROR!!!'
        elif self.eval_mode.startswith('location_choice'):
            try:
                generated_text = self.get_location_choice_answer(inputs)
            except:
                traceback.print_exc()
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
            merged_image = merge_images(inputs['images'])
            text_prompt = '\n' + temple_txt + inputs['above_content'] + '\n' + inputs['below_content']
            conv = conv_llava_v1.copy()
            text_prompt = text_prompt + "\n<image>\n" + temple_img
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=merged_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        
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
            images = load_image(inputs['image'])
            text_prompt = temple_txt + inputs['paragraphs']
            conv = conv_llava_v1.copy()
            text_prompt = text_prompt + "\n<image>\n" + temple_img
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        
    def get_fake_news_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'text': 
                'image': 
            }
        """  
        temple_img = "Does the above news article with image and text contain fake content? You only need to answer yes or no."
        if self.support_multi_image:
            raise NotImplementedError
        else:
            images = load_image(inputs['image'])
            text_prompt = inputs['text'] + "\n<image>\n" + temple_img
            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
            images = load_image(inputs['image'])
            text_prompt = "\n<image>\n"+ inputs['question']
            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=2048, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        
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
            images = load_image(inputs['image'])
            text_prompt = "\n<image>\n"+ inputs['question']
            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=2048, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
          
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
            images = load_image(inputs['image'])
            text_prompt = "\n<image>\n"+ inputs['question']
            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
          
          
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
            images = load_image(inputs['image'])
            text_prompt = "\n<image>\n"+ inputs['question']
            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
    def get_size_choice_answer(self, inputs: dict) -> str:
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
            images = load_image(inputs['image'])
            text_prompt = "\n<image>\n"+inputs['question']
            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text 
    def get_location_choice_answer(self, inputs: dict) -> str:
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
            images = load_image(inputs['image'])
            text_prompt = "\n<image>\n"+inputs['question']
            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text 
            
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
            question = '''
            \nHere are three images divided by black lines. The first image is a color picture containing a camouflaged creature, and the following two images are black and white mask images that outline the camouflaged creature. Please select the mask image that best matches the camouflaged creature in the first image.  The two mask images are labeled as ['A', 'B']. You need to only respond with the label of the image that is the correct match,donot give any explaation.\n
            '''
            merged_image = merge_images(inputs['image_list'])
            text_prompt = question+"\n<image>\n"

            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=merged_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        
    def get_size_compare_answer(self, inputs: dict) -> str:
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
            question = '''
            In the following two images divided by blcak linse, each contains a camouflaged creature. Please compare the relative area size that each camouflaged creature occupies in its respective image, and identify which creature occupies a larger relative area within the image. The comparison is based on the proportion of the area that the creature occupies in the image, not the actual size of the creature itself.The two images are labeled as ['A', 'B']. You need to only respond with the label of the image that is the correct match,donot give any explaation.\n
            '''
            merged_image = merge_images(inputs['image_list'])
            print(inputs['image_list'])
            text_prompt = question+"\n<image>\n"

            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=merged_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(generated_text)
            return generated_text
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
            merged_image = merge_images(inputs['image_list'])
            print(inputs['image_list'])
            text_prompt = inputs['question']+"\n<image>\n"

            conv = conv_llava_v1.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            text_prompt = conv.get_prompt()
            inputs = self.processor(text=text_prompt, images=merged_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(generated_text)
            return generated_text
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)