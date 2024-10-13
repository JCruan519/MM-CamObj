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
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.utils import is_flash_attn_2_available

class LlavaNext():
    support_multi_image = False
    merged_image_files = []
    def __init__(self, model_path:str="llava-hf/llava-v1.6-vicuna-7b-hf", eval_mode=None) -> None:
        """Llava model wrapper

        Args:
            model_path (str): Llava model name, e.g. "liuhaotian/llava-v1.5-7b" or "llava-hf/vip-llava-13b-hf"
        """
        self.model_path = model_path
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation).eval()
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        # self.processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True) 
        # self.model.resize_token_embeddings(len(self.processor.tokenizer))
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
            text_prompt = '\n' + temple_txt + inputs['above_content'] + '\n' + inputs['below_content'] + '\n' + temple_img
            if "mistral" in self.model_path:
                text_prompt = "[INST] <image>\n{} [/INST]".format(text_prompt)
            elif "vicuna" in self.model_path:
                text_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:".format(text_prompt)
            else:
                text_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:".format(text_prompt)
                # text_prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{}<|im_end|><|im_start|>assistant\n".format(text_prompt)
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
            text_prompt = temple_txt + inputs['paragraphs'] + '\n' + temple_img
            if "mistral" in self.model_path:
                
                text_prompt = "[INST] <image>\n{} [/INST]".format(text_prompt)
            elif "vicuna" in self.model_path:
                text_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:".format(text_prompt)

            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
             
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
    