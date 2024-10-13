"""pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118"""
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from io import BytesIO
from typing import List
from utils import merge_images, load_image
import traceback

class CogVLM():
    support_multi_image = False
    def __init__(self, model_path:str="llava-hf/llava-1.5-7b-hf", eval_mode=None) -> None:
        """
        Args:
            model_id (str): CogVLM model name, e.g. "THUDM/cogvlm-chat-hf"
        """
        self.model_id = model_path
        self.tokenizer = LlamaTokenizer.from_pretrained("/mnt/share/models/huggingface/rjc/models/lmsys/vicuna-7b-v1.5")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
        self.eval_mode = eval_mode
    
    def __call__(self, inputs: dict) -> str:
        if self.eval_mode.startswith('single_choice'):
            # try:
            generated_text = self.get_single_choice_anwser(inputs)
            # except:
                # return 'ERROR!!!'
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
            text_prompt = temple_txt + inputs['above_content'] + '\n' + inputs['below_content']
            text_prompt = text_prompt + temple_img
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=text_prompt, history=[], images=[merged_image])  # chat mode

            # Generate
            generated_text = self.get_parsed_output(inputs)
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
            text_prompt = text_prompt + temple_img
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=text_prompt, history=[], images=[images])  # chat mode
            # Generate
            generated_text = self.get_parsed_output(inputs)
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
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=text_prompt, history=[], images=[images])  # chat mode
            # Generate
            generated_text = self.get_parsed_output(inputs)
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
            text_prompt = inputs['question'] + "\n<image>\n"
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=text_prompt, history=[], images=[images])  # chat mode
            # Generate
            generated_text = self.get_parsed_output(inputs)
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
            text_prompt = inputs['question'] + "\n<image>\n"
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=text_prompt, history=[], images=[images])  # chat mode
            # Generate
            generated_text = self.get_parsed_output(inputs)
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
            text_prompt = inputs['question'] + "\n<image>\n"
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=text_prompt, history=[], images=[images])  # chat mode
            # Generate
            generated_text = self.get_parsed_output(inputs)
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
            merged_image = merge_images(inputs['images_list'])
            text_prompt = inputs['question'] + "\n<image>\n"
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=text_prompt, history=[], images=[merged_image])  # chat mode
            # Generate
            generated_text = self.get_parsed_output(inputs)
            return generated_text

    def get_parsed_output(self, inputs):
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        # gen_kwargs = {"max_length": 2048, "do_sample": False, 'no_repeat_ngram_size': 3, 'early_stopping': True}
        gen_kwargs = {"max_new_tokens": 128, "do_sample": False, 'no_repeat_ngram_size': 3, 'early_stopping': True}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output