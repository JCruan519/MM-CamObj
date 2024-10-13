import requests
import torch
import regex as re
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BatchFeature
from typing import List
from utils import merge_images, load_images, load_image
import traceback

class Kosmos2():
    support_multi_image = False
    def __init__(self, model_path:str="microsoft/kosmos-2-patch14-224", eval_mode=None) -> None:
        """
        Args:
            model_path (str): Kosmos2 model name, e.g. "microsoft/kosmos-2-patch14-224"
        """
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        self.eval_mode = eval_mode

        
    def __call__(self, inputs: List[dict]) -> str:
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
        elif self.eval_mode.startswith('VQA') or self.eval_mode.startswith('image_cap'):
            try:
                generated_text = self.get_general_answer(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('mask_FT'):
            try:
                generated_text = self.get_mask_FT_answer(inputs)
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
        additional_prompt = ""
        if self.support_multi_image:
            raise NotImplementedError
        else:
            # prompt = "What is in the image?\n"
            input_image = load_image(inputs['image'])
            text_prompt = inputs['question']
            text_prompt1 = text_prompt.split('Question:\n ')[0]
            text_prompt2 = text_prompt.split('Question:\n ')[-1]
            
            text_prompt = "<grounding> Question:"+ text_prompt1 + text_prompt2 +" Answer:"
            print(text_prompt)
            print("#"*10)
            inputs = self.processor(text=text_prompt, images=input_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
            new_generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0]
            print(generated_text.strip(" \n"))
            return generated_text.strip(" \n")

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
            # prompt = "What is in the image?\n"
            input_image = load_image(inputs['image'])
            text_prompt = inputs['question']
            text_prompt = text_prompt.split('Question:\n ')[-1]
            text_prompt = "Question:"+ text_prompt +" Answer:"
            print(text_prompt)
            inputs = self.processor(text=text_prompt, images=input_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=2048,
            )
            new_generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0]
            print(generated_text.strip(" \n"))
            return generated_text.strip(" \n")


    def get_bbox_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'question': 
                'image': 
            }
        """
        additional_prompt = "\nYour answer must in the format of [x1,y1,x2,y2]"
        if self.support_multi_image:
            raise NotImplementedError
        else:
            input_image = load_image(inputs['image'])
            text_prompt = inputs['question']
            text_prompt = text_prompt.split('Here is the question:')[-1]
            text_prompt = "<grounding> Question:" + text_prompt +" Answer:"
            print(text_prompt)
            inputs = self.processor(text=text_prompt, images=input_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
            new_generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip(" \n")

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
            input_image = load_image(inputs['image'])
            text_prompt = inputs['question']
            text_prompt = "<grounding> Question:" + text_prompt +" Answer:"
            
            inputs = self.processor(text=text_prompt, images=input_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
            new_generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0]
            answer = generated_text.strip(" \n")
            print(f"answer: {answer}")
            return generated_text.strip(" \n")

            
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
            merged_image = merge_images(inputs['image_list'])
            text_prompt = inputs['question']
            text_prompt = "<grounding> Question:" + text_prompt +" Answer:"
            
            inputs = self.processor(text=text_prompt, images=merged_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
            new_generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip(" \n")


   
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
            text_prompt = inputs['question']
            text_prompt = "<grounding> Question:" + text_prompt +" Answer:"
            
            inputs = self.processor(text=text_prompt, images=merged_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
            new_generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip(" \n")
    

    def process_interleaved_example(self, prompt, images, placeholder="<i>", num_image_tokens=64, add_special_tokens=True, add_eos_token=False, return_tensors=None):
        processor = self.processor

        first_image_token_id = processor.tokenizer.unk_token_id + 1

        image_input_ids = [processor.tokenizer.convert_tokens_to_ids(processor.boi_token)] + list(range(first_image_token_id, num_image_tokens + first_image_token_id)) + [processor.tokenizer.convert_tokens_to_ids(processor.eoi_token)]
        image_attention_mask = [1] * len(image_input_ids)
        # `-2`: not including `boi` and `eoi`
        image_embeds_position_mask = [0] + [1] * (len(image_input_ids) - 2) + [0]

        import re
        components = re.split(rf"({placeholder})", prompt)

        outputs = {"input_ids": [], "attention_mask": [], "image_embeds_position_mask": []}
        for component in components:
            if component != "<i>":
                # add text tokens: no special tokens -> add them at the end
                encoded = processor(text=component, add_special_tokens=False)
                for key in ["input_ids", "attention_mask"]:
                    outputs[key].extend(encoded[key])
                outputs["image_embeds_position_mask"].extend([0] * len(encoded["input_ids"]))
            else:
                # add tokens to indicate image placeholder
                outputs["input_ids"].extend(image_input_ids)
                outputs["attention_mask"].extend(image_attention_mask)
                outputs["image_embeds_position_mask"].extend(image_embeds_position_mask)

        if add_special_tokens:
            outputs["input_ids"] = [processor.tokenizer.bos_token_id] + outputs["input_ids"] + ([processor.tokenizer.eos_token_id] if add_eos_token else [])
            outputs["attention_mask"] = [1] + outputs["attention_mask"] + ([1] if add_eos_token else [])
            outputs["image_embeds_position_mask"] = [0] + outputs["image_embeds_position_mask"] + ([0] if add_eos_token  else [])

        outputs["pixel_values"] = processor.image_processor(images).pixel_values

        for k in ["input_ids", "attention_mask", "image_embeds_position_mask"]:
            outputs[k] = [outputs[k]]
        outputs = BatchFeature(data=outputs,tensor_type=return_tensors)

        return outputs
    