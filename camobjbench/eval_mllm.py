import json
import regex as re
import os
import datasets
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.join(os.getcwd(), '..')))
from mllm_eval import MLLM_Models
from utils import load
from typing import List
import re
import pandas as pd
import random
import torch
import argparse  # 导入argparse模块
from FlagEmbedding.visual.modeling import Visualized_BGE
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def eval_easy_VQA(benchmark_data, benchmark_root_path, model):
    correct = 0
    format_error = 0
    all_true_labels = []
    all_model_answers = []
    total_questions = len(benchmark_data)

    answer_list = []

    for line in tqdm(benchmark_data):
        result_dict = {
            'question_id' : line['question_id'],
            'image_id' : int(line['id'].split("_")[0]),
            'true_answer' : line['answer'],
            'model_answer' : ''
        }

        true_answer = line['answer']
        question = line['text']
        image_path = line['image']
        all_true_labels.append(true_answer)

        inputs = {
            'question': question,
            'image': benchmark_root_path + '/' + image_path
        }

        # get raw answer
        model_answer = model(inputs)
        if model_answer.startswith('A'):
            model_answer = "A"
        elif model_answer.startswith('B'):
            model_answer = "B"
        elif model_answer.startswith('C'):
            model_answer = "C"
        elif model_answer.startswith('D'):
            model_answer = "D"

        print(model_answer)

        if model_answer not in ['A', 'B', 'C', 'D']:
            model_answer = 'A'
            format_error += 1
        if model_answer == true_answer:
                correct += 1
        result_dict['model_answer'] = model_answer    
        all_model_answers.append(model_answer)
        answer_list.append(result_dict)

    # 计算性能指标
    accuracy = accuracy_score(all_true_labels, all_model_answers)
    precision = precision_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels, all_model_answers, average='macro', zero_division=0)

    # 打印结果
    print(f"Correct Questions: {correct}")
    print(f"Total Questions: {total_questions}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Wrong formart answer : {format_error}")

    # 返回结果字典和性能指标
    return {
        'performance_metrics': {
            'accuracy': correct / total_questions,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'answer_list':answer_list
    }


def eval_hard_VQA(benchmark_data, benchmark_root_path, model):
    correct = 0
    format_error = 0
    all_true_labels = []
    all_model_answers = []
    total_questions = len(benchmark_data)

    answer_list = []

    for line in tqdm(benchmark_data):
        resault_dict = {
            'question_id' : line['question_id'],
            'image_id' : int(line['id'].split("_")[0]),
            'true_answer' : line['answer'],
            'model_answer' : ''
        }

        true_answer = line['answer']
        question = line['text']
        image_path = line['image']
        all_true_labels.append(true_answer)

        inputs = {
            'question': question,
            'image': benchmark_root_path + '/' + image_path
        }

        # get raw answer
        model_answer = model(inputs)
        if model_answer.startswith('A'):
            model_answer = "A"
        elif model_answer.startswith('B'):
            model_answer = "B"
        elif model_answer.startswith('C'):
            model_answer = "C"
        elif model_answer.startswith('D'):
            model_answer = "D"

        print(model_answer)

        if model_answer not in ['A', 'B', 'C', 'D']:
            model_answer = 'A'
            format_error += 1
        if model_answer == true_answer:
                correct += 1
        resault_dict['model_answer'] = model_answer    
        all_model_answers.append(model_answer)
        answer_list.append(resault_dict)

    # 计算性能指标
    accuracy = accuracy_score(all_true_labels, all_model_answers)
    precision = precision_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels, all_model_answers, average='macro', zero_division=0)

    # 打印结果
    print(f"Correct Questions: {correct}")
    print(f"Total Questions: {total_questions}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Wrong formart answer : {format_error}")

    # 返回结果字典和性能指标
    return {
        'performance_metrics': {
            'accuracy': correct / total_questions,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'answer_list':answer_list
    }

def calculate_iou(boxA, boxB):
    # 计算两个边界框的交集坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集的面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个边界框的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并集的面积
    unionArea = boxAArea + boxBArea - interArea

    # 计算IoU
    iou = interArea / float(unionArea) if unionArea > 0 else 0

    return iou

def calculate_miou(bbox_true, bbox_model):
    # 确保输入的边界框列表长度相同
    if len(bbox_true) != len(bbox_model):
        raise ValueError("The length of true and model bounding boxes must be the same.")

    # 计算所有边界框的IoU
    ious = [calculate_iou(true, model) for true, model in zip(bbox_true, bbox_model)]

    # 计算平均IoU
    mean_iou = sum(ious) / len(ious)

    return round(mean_iou, 4)

def extract_bbox_from_text(model_output):
    bbox_pattern = re.compile(r'\b(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\b')
    matches = bbox_pattern.findall(model_output)
    if matches:
        return [float(coord) for coord in matches[0]]
    else:
        return [0, 0, 0, 0]
    

def eval_bbox_without_name(benchmark_data, benchmark_root_path, model):
    correct = 0
    format_error = 0
    all_true_labels = []
    all_model_answers = []
    total_questions = len(benchmark_data)

    answer_list = []

    prompt = '''
    You are an expert in camouflage biological monitoring. Your task is to detect camouflaged creatures in pictures and mark its location with a BoundingBox.
    **IMPORTANT**
    1.Your answer should be the BoundingBox of the camouflaged creatures
    2.Don't interpret your answer in any way
    '''

    for line in tqdm(benchmark_data):
        resault_dict = {
            'question_id' : line['question_id'],
            'image_id' : int(line['id'].split("_")[0]),
            'true_answer' : line['answer'],
            'model_answer' : ''
        }

        true_answer = line['answer']
        question = line['text']
        image_path = line['image']
        all_true_labels.append(true_answer)

        question = prompt+question 
        inputs = {
            'question': question,
            'image': benchmark_root_path + '/' + image_path
        }

        # get raw answer
        model_answer = model(inputs)
        print(model_answer)
        model_answer = extract_bbox_from_text(model_answer)
        if model_answer == [0,0,0,0]:
            format_error+=1
        print(model_answer)


        resault_dict['model_answer'] = model_answer    
        all_model_answers.append(model_answer)
        answer_list.append(resault_dict)

    miou = calculate_miou(all_true_labels,all_model_answers)
    print(f'miou: {miou}')
    print(f'formart_error: {format_error}')
    # 返回结果字典和性能指标
    return {
        'performance_metrics': {
            'miou':miou,
            'formart_error':format_error
        },
        'answer_list':answer_list
    }


def eval_count_choice(benchmark_data, benchmark_root_path, model):

    def convert_number_words(sentence):
        # 创建一个字典映射数字单词到数字
        number_map = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9"
        }
        
        # 将句子分割为单词
        words = sentence.split()
        
        # 遍历每个单词，替换数字单词为数字
        converted_words = [number_map[word] if word in number_map else word for word in words]
        
        # 将单词列表重新拼接为句子
        converted_sentence = ' '.join(converted_words)
        
        return converted_sentence
    
    correct = 0
    format_error = 0
    all_true_labels = []
    all_model_answers = []
    total_questions = len(benchmark_data)

    answer_list = []

    prompt = '''
    You are an expert in camouflage biological monitoring. Your task is to detect camouflaged creatures in pictures and answer a multiple-choice question containing a text description and four choices. Please choose the answer you think is most appropriate from the four choices [A, B, C, D]. If you are not sure of the answer, please still choose the answer you think is most likely to be correct.
    **IMPORTANT**
    1.Your answer should be just one letter in [A, B, C, D]
    2.Don't interpret your answer in any way
    '''

    for line in tqdm(benchmark_data):
        resault_dict = {
            'question_id' : line['question_id'],
            'image_id' : int(line['id'].split("_")[0]),
            'true_answer' : line['answer'],
            'model_answer' : '',
            'raw_model_answer': ''
        }

        true_answer = line['answer']
        question = line['text']
        image_path = line['image']

        question = prompt + question 

        inputs = {
            'question': question,
            'image': benchmark_root_path + '/' + image_path
        }

        # get raw answer
        model_answer = model(inputs)
        resault_dict['raw_model_answer'] = model_answer
        if model_answer.startswith('A'):
            model_answer = "A"
        elif model_answer.startswith('B'):
            model_answer = "B"
        elif model_answer.startswith('C'):
            model_answer = "C"
        elif model_answer.startswith('D'):
            model_answer = "D"


        if model_answer not in ['A', 'B', 'C', 'D']:
            # model_answer = 'A'
            # format_error += 1
            options = line['text'].split('\n')[1:]
            answer_key = line['answer']
            answer_number = None
            for option in options:
                if option.startswith(answer_key):
                    # 提取选项中的数字部分
                    answer_number = option.split('. ')[1] 
            true_answer = answer_number
            resault_dict['true_answer'] = true_answer
            model_answer = convert_number_words(model_answer)
             # 正则表达式匹配0到100之间的整数
            pattern = r'\b([1-9]?\d|100)\b'

            # 使用正则表达式搜索文本
            match = re.search(pattern, model_answer)
            if match:
                # 如果找到匹配项，提取数字
                model_answer = str(match.group(0))
            else:
                model_answer = "0"
                format_error += 1
        if model_answer == true_answer:
            correct += 1
        resault_dict['model_answer'] = model_answer    
        
        all_true_labels.append(true_answer)
        all_model_answers.append(model_answer)
        answer_list.append(resault_dict)

    # 计算性能指标
    accuracy = accuracy_score(all_true_labels, all_model_answers)
    precision = precision_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels, all_model_answers, average='macro', zero_division=0)

    # 打印结果
    print(f"Correct Questions: {correct}")
    print(f"Total Questions: {total_questions}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Wrong formart answer : {format_error}")

    # 返回结果字典和性能指标
    return {
        'performance_metrics': {
            'accuracy': correct / total_questions,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'answer_list':answer_list
    }


def eval_mask_match(benchmark_data, benchmark_root_path, model):
    correct = 0
    format_error = 0
    all_true_labels = []
    all_model_answers = []
    total_questions = len(benchmark_data)

    answer_list = []

    prompt = '''
    '''

    for line in tqdm(benchmark_data):
        resault_dict = {
            'question_id' : line['question_id'],
            'image_id' : int(line['id'].split("_")[0]),
            'true_answer' : line['answer'],
            'model_answer' : ''
        }

        true_answer = line['answer']
        question = line['text']
        image_path_list = line['image']
        all_true_labels.append(true_answer)

        question = question + prompt

        
        inputs = {
            'question': question,
            'image_list':[benchmark_root_path + '/' + image for image in image_path_list]
        }

        # get raw answer
        model_answer = model(inputs)
        if model_answer.startswith('A'):
            model_answer = "A"
        elif model_answer.startswith('B'):
            model_answer = "B"
        elif model_answer.startswith('C'):
            model_answer = "C"
        elif model_answer.startswith('D'):
            model_answer = "D"

        print(model_answer)

        if model_answer not in ['A', 'B', 'C', 'D']:
            model_answer = 'A'
            format_error += 1
        if model_answer == true_answer:
                correct += 1
        resault_dict['model_answer'] = model_answer    
        all_model_answers.append(model_answer)
        answer_list.append(resault_dict)

    # 计算性能指标
    accuracy = accuracy_score(all_true_labels, all_model_answers)
    precision = precision_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels, all_model_answers, average='macro', zero_division=0)

    # 打印结果
    print(f"Correct Questions: {correct}")
    print(f"Total Questions: {total_questions}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Wrong formart answer : {format_error}")

    # 返回结果字典和性能指标
    return {
        'performance_metrics': {
            'accuracy': correct / total_questions,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'answer_list':answer_list
    }


def eval_mask_FT(benchmark_data, benchmark_root_path, model):
    correct = 0
    format_error = 0
    all_true_labels = []
    all_model_answers = []
    total_questions = len(benchmark_data)

    answer_list = []

    for line in tqdm(benchmark_data):
        resault_dict = {
            'question_id' : line['question_id'],
            'image_id' : int(line['id'].split("_")[0]),
            'true_answer' : line['answer'],
            'model_answer' : ''
        }

        true_answer = line['answer']
        question = line['text']
        image_path_list = line['image']
        all_true_labels.append(true_answer)

        question = question

        inputs = {
            'question': question,
            'image_list':[benchmark_root_path + '/' + image for image in image_path_list]
        }

        # get raw answer
        model_answer = model(inputs)
        print(model_answer)
        if "True" in model_answer or "true" in model_answer:
            model_answer = "True"
        elif "False" in model_answer or "false" in model_answer:
            model_answer = "False"
        else:
            model_answer = "False"
            format_error+=1

        print(model_answer)

        if model_answer == true_answer:
                correct += 1
        resault_dict['model_answer'] = model_answer    
        all_model_answers.append(model_answer)
        answer_list.append(resault_dict)

    # 计算性能指标
    accuracy = accuracy_score(all_true_labels, all_model_answers)
    precision = precision_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels, all_model_answers, average='macro', zero_division=0)

    # 打印结果
    print(f"Correct Questions: {correct}")
    print(f"Total Questions: {total_questions}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Wrong formart answer : {format_error}")

    # 返回结果字典和性能指标
    return {
        'performance_metrics': {
            'accuracy': correct / total_questions,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'answer_list':answer_list
    }


def eval_image_cap(benchmark_data, benchmark_root_path, model, model_bge):

    def compute_clip_score(model_answer, true_answer, model_bge):
        model_answer_emb = model_bge.encode(text=model_answer)
        true_answer_emb = model_bge.encode(text=true_answer)
        similarity = model_answer_emb @ true_answer_emb.T
        return similarity.item()

    answer_list = []
    clip_score_list = []

    for item in tqdm(benchmark_data):
        result_dict = {
            'image' : item['image'],
            'true_answer' : item['answer'],
            'model_answer' : '',
            'clip_score': -1,
        }

        inputs = {
            'question': item['question'],
            'image': benchmark_root_path + '/' + item['image']
        }

        # get raw answer
        model_answer = model(inputs)

        result_dict['model_answer'] = model_answer    
        result_dict['clip_score'] = compute_clip_score(model_answer, item['answer'], model_bge)    
        clip_score_list.append(result_dict['clip_score'])

        print(model_answer, result_dict['clip_score'])

        answer_list.append(result_dict)

    # 打印结果
    print(f"Mean Clip Score: {np.mean(clip_score_list)}")

    # 返回结果字典和性能指标
    return {
        'mean_clip_score': np.mean(clip_score_list),
        'answer_list': answer_list
    }





import os
import random
import pandas as pd
import json

def main(
    model_name: str,
    model_path: str,
    dataset_path: str,
    results_dir: str,
    img_path = None,
    eval_mode: str='single_choice',
    seed = 42
):
    random.seed(seed)
    # 加载benchmark数据
    benchmark_data = load(dataset_path)
    benchmark_root_path = os.path.dirname(os.path.abspath(dataset_path)) if img_path == 'None' else img_path
    
    # 初始化模型
    if model_name == "random":
        model = None
    else:
        model = MLLM_Models(model_name, model_path, eval_mode)

    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)

    # 调用验证函数进行验证
    if eval_mode.startswith('easy_VQA'):
        results_dict = eval_easy_VQA(benchmark_data, benchmark_root_path, model)
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

    elif eval_mode.startswith('image_cap'):
        model_bge = Visualized_BGE(
            model_name_bge='BAAI/bge-base-en-v1.5', 
            model_weight='BAAI/bge-visualized/Visualized_base_en_v1.5.pth'
        ).cuda()
        results_dict = eval_image_cap(benchmark_data, benchmark_root_path, model, model_bge)
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        del model
        torch.cuda.empty_cache()
        return results_dict
    elif eval_mode.startswith('hard_VQA'):
        results_dict = eval_hard_VQA(benchmark_data, benchmark_root_path, model)
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        del model
        torch.cuda.empty_cache()
        return results_dict
    elif eval_mode.startswith('bbox_without_name'):
        results_dict = eval_bbox_without_name(benchmark_data, benchmark_root_path, model)
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        del model
        torch.cuda.empty_cache()
        return results_dict
    elif eval_mode.startswith('count_choice'):
        results_dict = eval_count_choice(benchmark_data, benchmark_root_path, model)
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        del model
        torch.cuda.empty_cache()
        return results_dict
    elif eval_mode.startswith('mask_match'):
        results_dict = eval_mask_match(benchmark_data, benchmark_root_path, model)
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        del model
        torch.cuda.empty_cache()
        return results_dict
    elif eval_mode.startswith('mask_FT'):
        results_dict = eval_mask_FT(benchmark_data, benchmark_root_path, model)
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        del model
        torch.cuda.empty_cache()
        return results_dict
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
    # 给这个解析对象添加命令行参数
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--img_path', type=str, default='None')
    parser.add_argument('--results_dir', type=str, default='None')
    parser.add_argument('--eval_mode', type=str, default='single_choice_1')
    args = parser.parse_args()  # 获取所有参数

    results_dir = args.results_dir + '/' + args.eval_mode + '/' + args.model_path.split('/')[-1]

    results = main(
        model_name=args.model_name,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        results_dir=results_dir,
        img_path=args.img_path,
        eval_mode=args.eval_mode,
        seed=42
    )