"""pip install transformers>=4.35.2 transformers_stream_generator torchvision tiktoken chardet matplotlib
""" 
import tempfile
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import List
from utils import merge_images, load_image
import re
import traceback

class QwenVL():
    support_multi_image = False
    merged_image_files = []
    def __init__(self, model_path:str="Qwen/Qwen-VL-Chat", eval_mode=None) -> None:
        """
        Args:
            model_path (str): Qwen model name, e.g. "Qwen/Qwen-VL-Chat"
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
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
            text_prompt = temple_txt + inputs['above_content'] + '\n' + inputs['below_content']
            text_prompt = text_prompt + temple_img
            inputs = self.prepare_prompt(inputs['images'], text_prompt)
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
            text_prompt = temple_txt + inputs['paragraphs']
            text_prompt = text_prompt + "\n<image>\n" + temple_img
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
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
            text_prompt = inputs['text'] + "\n<image>\n" + temple_img
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
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
            text_prompt = inputs['question']
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
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
            text_prompt = inputs['question']
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
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
            text_prompt = inputs['question']
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
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
            text_prompt = inputs['question']
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
            return generated_text 
          
          
    def parse_and_normalize_bbox(self,input_str):
        # 正则表达式匹配括号内的坐标
        bbox_pattern = re.compile(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>')
        
        # 搜索匹配项
        match = bbox_pattern.search(input_str)
        if match:
            # 提取坐标
            x1, y1, x2, y2 = map(int, match.groups())
            
            # 假设图像的宽度和高度分别为1000（这需要根据实际情况调整）
            image_width = 1000
            image_height = 1000
            
            # 归一化坐标
            normalized_bbox = [
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height
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
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
            print(generated_text)
            return str(self.parse_and_normalize_bbox(generated_text)) 
        
    def get_fake_news_answer(self, inputs: dict) -> str:
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
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
            print(generated_text)
            return str(self.parse_and_normalize_bbox(generated_text))
          
          
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
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
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
        prompt = "\nHere are three images. The first image is a picture containing a camouflaged creature, and the following two images are mask images that outline the camouflaged creature. Please select the mask image that best matches the camouflaged creature in the first image. The two mask images are labeled as ['A', 'B']. You need to respond with the label of the image that is the correct match.\n**IMPORTANT**1.Your answer should be just one letter in [A, B]2.Don't interpret your answer in any way"
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image_list'], prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
            return generated_text   
          
          
    def get_size_compare_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image_list': 
            }
        """
        prompt = "In the following two images, each contains a camouflaged creature. Please compare the relative area size that each camouflaged creature occupies in its respective image, and identify which creature occupies a larger relative area within the image. The comparison is based on the proportion of the area that the creature occupies in the image, not the actual size of the creature itself.\n**IMPORTANT**1.Your answer should be just one letter in [A, B]2.Don't interpret your answer in any way"
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image_list'], prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
            return generated_text   
    def get_mask_FT_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image_list': 
            }
        """
        # prompt = "In the following two images, each contains a camouflaged creature. Please compare the relative area size that each camouflaged creature occupies in its respective image, and identify which creature occupies a larger relative area within the image. The comparison is based on the proportion of the area that the creature occupies in the image, not the actual size of the creature itself.\n**IMPORTANT**1.Your answer should be just one letter in [A, B]2.Don't interpret your answer in any way"
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = inputs['question']
            text_prompt = text_prompt
            inputs = self.prepare_prompt(inputs['image_list'],text_prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
            print(generated_text)
            return generated_text 

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        true_image_links = []
        for i, image_link in enumerate(image_links):
            if isinstance(image_link, str):
                true_image_links.append(image_link)
            elif isinstance(image_link, Image.Image):
                image_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                image_file.close()
                image_link.save(image_file.name)
                self.merged_image_files.append(image_file.name)
                true_image_links.append(image_file.name)
            else:
                raise NotImplementedError
        image_links = true_image_links
        input_list = []
        for i, image_link in enumerate(image_links):
            input_list.append({'image': image_link})
        input_list.append({'text': text_prompt})
        query = self.tokenizer.from_list_format(input_list)
        return query
    

    def get_parsed_output(self, query):
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def __del__(self):
        for image_file in self.merged_image_files:
            os.remove(image_file)