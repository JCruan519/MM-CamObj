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



def eval_single_choice(benchmark_data, benchmark_root_path, model):
    def generate_instruction_string(img_list_len):
        labels = ['A', 'B', 'C', 'D']
        chosen_labels = labels[:img_list_len]
        labels_str = ', '.join(chosen_labels)
        return f"Please choose the most appropriate image to insert between the two paragraphs above. The given {img_list_len} images are labeled {labels_str}, separated by black lines. You only need to respond with the letter of the chosen image."

    def find_answer_in_string(text):
        # Define the regular expression pattern to match standalone letters A, B, C, D
        pattern = r'(?<![\w])([A-D])(?![\w])'

        # Search for the pattern in the text
        match = re.search(pattern, text.upper())

        if match:
            return match.group(1)
        else:
            return None

    correct = 0
    results = []

    for _, row in tqdm(benchmark_data.iterrows(), desc=f'Evaluating'):
        images_list = []
        # Iterate over each column in the row
        for column_name in benchmark_data.columns:
            # Check if the column name starts with 'img'
            if column_name.startswith('img') and not column_name.startswith('url') and not pd.isna(row[column_name]):
                images_list.append(benchmark_root_path + '/' + row[column_name])

        # Generate the instruction string
        temple_img = generate_instruction_string(len(images_list))

        if pd.isna(row['below_content']): row['below_content'] = ''
        inputs = {
            'above_content': row['above_content'],
            'below_content': row['below_content'],
            'images': images_list,
            'temple_img': temple_img
        }

        # Get the model's raw answer
        raw_answer = model(inputs)
        model_answer = find_answer_in_string(raw_answer)
        true_label = row['Answer']

        # Check if the model's answer is correct
        if model_answer == true_label:
            correct += 1

        # Append the results to the list
        result_row = row.to_dict()
        result_row['raw_answer'] = raw_answer
        result_row['model_answer'] = model_answer
        results.append(result_row)

    # Calculate the accuracy
    accuracy = correct / len(benchmark_data)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    return results_df, accuracy



from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_flow_insert(benchmark_data, benchmark_root_path, model):

    def find_confidence_in_string(text):
        # Define the regular expression pattern to match numbers between 0 and 100
        pattern = r'(?:100(?:\.0{1,})?|(?:[1-9]?\d|0)(?:\.\d{1,})?)'

        # Search for the pattern in the text
        match = re.search(pattern, text)

        if match:
            return float(match.group())
        else:
            return None

    correct_matches = 0
    total_matches = 0
    all_true_labels = []
    all_model_answers = []
    news_item_results = []

    for domain, keywords in tqdm(benchmark_data.items(), desc="Domains"):
        print(f"domain: {domain}")
        for keyword, content in tqdm(keywords.items(), desc="Keywords"):
            print(f"keyword: {keyword}")
            image_database = content['imagedatabase']

            for news_item in tqdm(content['news_text'], desc="News Items"):
                selected_images = set()
                paragraph_content = news_item['content']
                model_answer = ["" for i in range(len(paragraph_content))]
                true_label = news_item['groundtruth']
                inputs_paragraphs = ''
                news_id = news_item['id']

                # 遍历每一个段落
                for index_paragraph, paragraph in enumerate(paragraph_content):
                    inputs_paragraphs += '\n' + paragraph
                    highest_confidence = -float('inf')
                    best_image_idx = None

                    # 遍历每一张图片
                    for index_img, img in enumerate(image_database):
                        if index_img in selected_images:
                            continue
                        inputs = {
                            'paragraphs': inputs_paragraphs,
                            'image': benchmark_root_path + '/' + img[0],
                        }

                        # Get the model's raw answer
                        raw_answer = model(inputs)
                        print(raw_answer)
                        confidence = find_confidence_in_string(raw_answer)

                        if confidence is not None:
                            if confidence > highest_confidence:
                                highest_confidence = confidence
                                best_image_idx = index_img

                    if best_image_idx is not None:
                        selected_images.add(best_image_idx)
                        model_answer[index_paragraph] = image_database[best_image_idx][0]

                all_true_labels.extend(true_label)
                all_model_answers.extend(model_answer)

                for model_answer_item, true_label_item in zip(model_answer, true_label):
                    if model_answer_item == true_label_item:
                        correct_matches += 1
                    total_matches += 1

                # 保存news_item的结果
                news_item_results.append({
                    'id': news_id,
                    'true_label': true_label,
                    'model_answer': model_answer
                })
            break
        break
    # 计算性能指标
    # accuracy = accuracy_score(all_true_labels, all_model_answers)
    # precision = precision_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    # recall = recall_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
    # f1 = f1_score(all_true_labels, all_model_answers, average='macro', zero_division=0)

    # # 打印结果
    # print(f"Correct Matches: {correct_matches}")
    # print(f"Total Matches: {total_matches}")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")

    # 返回结果字典和性能指标
    return {
        'performance_metrics': {
            'accuracy': correct_matches / total_matches,
            # 'precision': precision,
            # 'recall': recall,
            # 'f1_score': f1
        },
        'news_item_results': news_item_results,
    }


from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import pandas as pd

def eval_fake_news(benchmark_data, benchmark_root_path, model, eval_mode):

    def check_news_status(text):
        # 定义匹配关键词的正则表达式，并使用 re.IGNORECASE 来忽略大小写
        yes_pattern = re.compile(r'\byes\b', re.IGNORECASE)
        true_pattern = re.compile(r'\btrue\b', re.IGNORECASE)

        # 在输入字符串中搜索模式
        yes_match = yes_pattern.search(text)
        true_match = true_pattern.search(text)

        # 如果同时找到 'yes' 和 'true'，返回 1，否则返回 0
        if yes_match or true_match:
            return 1
        else:
            return 0

    results = []
    gt_label_list = []
    model_answer_list  = []

    if 'dgm4' in eval_mode:
        benchmark_data = random.sample(benchmark_data, 500)
        for news_item in tqdm(benchmark_data, desc=f'Evaluating'):
            inputs = {
                'text': news_item["text"],
                'image': benchmark_root_path + '/' + news_item["image"],
            }
            is_fake_news = 0 if news_item["fake_cls"] == "orig" else 1
            gt_label_list.append(is_fake_news)

            if model is None:
                raw_answer = model_answer = random.choice([0, 1])
            else:
                raw_answer = model(inputs)
                model_answer = check_news_status(raw_answer)

            model_answer_list.append(model_answer)

            results.append({
                "text": news_item["text"],
                "image": news_item["image"],
                "fake_cls": news_item["fake_cls"],
                "is_fake_news": is_fake_news,
                "raw_answer": raw_answer,
                "model_answer": model_answer
            })
        accuracy = accuracy_score(gt_label_list, model_answer_list)
        precision = precision_score(gt_label_list, model_answer_list)
        recall = recall_score(gt_label_list, model_answer_list)
        f1 = f1_score(gt_label_list, model_answer_list)

        print(f"True Mode - Accuracy: {accuracy}")
        print(f"True Mode - Precision: {precision}")
        print(f"True Mode - Recall: {recall}")
        print(f"True Mode - F1 Score: {f1}")
    else:
        gt_simple_mode_label_list = []
        model_simple_mode_answer_list  = []
        gt_difficult_mode_label_list = []
        model_difficult_mode_answer_list  = []
        # Iterate over each domain
        for domain, keywords in benchmark_data.items():
            # Iterate over each keyword
            for keyword, content in keywords.items():
                # Iterate over each news item
                for news_item in content['news_text']:
                    ## Test True news

                    true_text = news_item['title']
                    for i, text in enumerate(news_item['content']):
                        true_text += '\n' + text
                    image = benchmark_root_path + '/' + news_item['images'][0]['img']  # 暂时先考虑只给第一张图片

                    inputs = {
                        'text': true_text[:1024],
                        'image': image,
                    }

                    is_fake_news = 0
                    gt_label_list.append(is_fake_news)

                    if model is None:
                        raw_answer = model_answer = random.choice([0, 1])
                    else:
                        raw_answer = model(inputs)
                        model_answer = check_news_status(raw_answer)

                    model_answer_list.append(model_answer)

                    results.append({
                        "id": news_item['id'],
                        "text": true_text,
                        "image": image,
                        "is_fake_news": is_fake_news,
                        "raw_answer": raw_answer,
                        "model_answer": model_answer
                    })

                    ## Test SimpleMode FakeNews

                    inputs = {
                        'text': news_item['json_respond_text']['SimpleMode']['FakeNews'][:1024],
                        'image': image,
                    }

                    is_fake_news = 1
                    gt_simple_mode_label_list.append(is_fake_news)

                    if model is None:
                        raw_answer = model_answer = random.choice([0, 1])
                    else:
                        raw_answer = model(inputs)
                        model_answer = check_news_status(raw_answer)

                    model_simple_mode_answer_list.append(model_answer)

                    results.append({
                        "id": news_item['id'],
                        "text": news_item['json_respond_text']['SimpleMode']['FakeNews'],
                        "image": image,
                        "is_fake_news": is_fake_news,
                        "raw_answer": raw_answer,
                        "model_answer": model_answer
                    })

                    ## Test DifficultMode FakeNews

                    inputs = {
                        'text': news_item['json_respond_text']['DifficultMode']['FakeNews'][:1024],
                        'image': image,
                    }

                    is_fake_news = 1
                    gt_difficult_mode_label_list.append(is_fake_news)

                    if model is None:
                        raw_answer = model_answer = random.choice([0, 1])
                    else:
                        raw_answer = model(inputs)
                        model_answer = check_news_status(raw_answer)

                    model_difficult_mode_answer_list.append(model_answer)

                    results.append({
                        "id": news_item['id'],
                        "text": news_item['json_respond_text']['DifficultMode']['FakeNews'],
                        "image": image,
                        "is_fake_news": is_fake_news,
                        "raw_answer": raw_answer,
                        "model_answer": model_answer
                    })

        # 合并标签和预测结果
        gt_combined_simple = gt_label_list + gt_simple_mode_label_list
        model_combined_simple = model_answer_list + model_simple_mode_answer_list

        gt_combined_difficult = gt_label_list + gt_difficult_mode_label_list
        model_combined_difficult = model_answer_list + model_difficult_mode_answer_list

        accuracy_simple = accuracy_score(gt_combined_simple, model_combined_simple)
        precision_simple = precision_score(gt_combined_simple, model_combined_simple)
        recall_simple = recall_score(gt_combined_simple, model_combined_simple)
        f1_simple = f1_score(gt_combined_simple, model_combined_simple)

        accuracy_difficult = accuracy_score(gt_combined_difficult, model_combined_difficult)
        precision_difficult = precision_score(gt_combined_difficult, model_combined_difficult)
        recall_difficult = recall_score(gt_combined_difficult, model_combined_difficult)
        f1_difficult = f1_score(gt_combined_difficult, model_combined_difficult)

        # 输出结果
        print(f"Combined Simple Mode - Accuracy: {accuracy_simple}")
        print(f"Combined Simple Mode - Precision: {precision_simple}")
        print(f"Combined Simple Mode - Recall: {recall_simple}")
        print(f"Combined Simple Mode - F1 Score: {f1_simple}")

        print(f"Combined Difficult Mode - Accuracy: {accuracy_difficult}")
        print(f"Combined Difficult Mode - Precision: {precision_difficult}")
        print(f"Combined Difficult Mode - Recall: {recall_difficult}")
        print(f"Combined Difficult Mode - F1 Score: {f1_difficult}")

    # 将结果转换为DataFrame
    df = pd.DataFrame(results)
    return df


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
def eval_size_choice(benchmark_data, benchmark_root_path, model):
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
            'image':benchmark_root_path + '/' + image_path 
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
def eval_location_choice(benchmark_data, benchmark_root_path, model):
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
            'image':benchmark_root_path + '/' + image_path 
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
    
def eval_bbox_with_name(benchmark_data, benchmark_root_path, model):
    correct = 0
    format_error = 0
    all_true_labels = []
    all_model_answers = []
    total_questions = len(benchmark_data)

    answer_list = []

    prompt = '''
    You are an expert in camouflage biological monitoring. Your task is to answer a question of detect camouflaged creature in a picture and mark its location with a BoundingBox.
    **IMPORTANT**
    1.Your answer should be the BoundingBox of the camouflaged creature
    2.Don't interpret your answer in any way
    3.If you are not sure where the camouflaged creature is, provide the bounding box where the creature is most likely to be.\n
    Here is the question:\n
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
def eval_count_num(benchmark_data, benchmark_root_path, model):

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
    You are an expert in camouflage biological monitoring. Your task is to detect camouflaged creatures in pictures and answer how many camouflaged creatures are there in the picture.
    **IMPORTANT**
    1.Your answer should be just one number in Arabic numerals [0~10].
    2.Don't interpret your answer in any way.

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
        all_true_labels.append(str(true_answer))

        question = prompt+question 

        inputs = {
            'question': question,
            'image': benchmark_root_path + '/' + image_path
        }

        # get raw answer
        model_answer = model(inputs)
        
        # 正则表达式匹配0到100之间的整数
        pattern = r'\b([1-9]?\d|100)\b'

        # 使用正则表达式搜索文本
        match = re.search(pattern, model_answer)

        if match:
            # 如果找到匹配项，提取数字
            model_answer = str(match.group(0))
        else:
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
        print(model_answer)


        if model_answer == str(true_answer):
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
def eval_size_compare(benchmark_data, benchmark_root_path, model):
    correct = 0
    format_error = 0
    all_true_labels = []
    all_model_answers = []
    total_questions = len(benchmark_data)

    answer_list = []

    # prompt = '''
    # You are an expert in camouflage biological monitoring. Your task is to detect camouflaged creatures in pictures and answer A multiple-choice question containing a text description and four choices. Please choose the answer you think is most appropriate from the four choices [A, B, C, D]. If you are not sure of the answer, please still choose the answer you think is most likely to be correct.
    # **IMPORTANT**
    # 1.Your answer should be just one letter in [A, B, C, D]
    # 2.Don't interpret your answer in any way
    # '''

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

    # prompt = '''
    # You are an expert in camouflage biological monitoring. Your task is to detect camouflaged creatures in pictures and answer A multiple-choice question containing a text description and four choices. Please choose the answer you think is most appropriate from the four choices [A, B, C, D]. If you are not sure of the answer, please still choose the answer you think is most likely to be correct.
    # **IMPORTANT**
    # 1.Your answer should be just one letter in [A, B, C, D]
    # 2.Don't interpret your answer in any way
    # '''

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
    if eval_mode.startswith('single_choice'):
        results_df, accuracy = eval_single_choice(benchmark_data, benchmark_root_path, model)

        # 保存结果为CSV文件
        csv_path = os.path.join(results_dir, 'results.csv')
        results_df.to_csv(csv_path, index=False)

        del model
        torch.cuda.empty_cache()
        print(accuracy)
        return results_df

    elif eval_mode.startswith('flow_insert'):
        results_dict = eval_flow_insert(benchmark_data, benchmark_root_path, model)

        # 保存结果为JSON文件
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        del model
        torch.cuda.empty_cache()
        return results_dict

    elif eval_mode.startswith('fake_news'):
        results_df = eval_fake_news(benchmark_data, benchmark_root_path, model, eval_mode)

        # 保存结果为CSV文件
        csv_path = os.path.join(results_dir, 'results.csv')
        results_df.to_csv(csv_path, index=False)

        del model
        torch.cuda.empty_cache()
        return results_df
    elif eval_mode.startswith('easy_VQA'):
        results_dict = eval_easy_VQA(benchmark_data, benchmark_root_path, model)
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

    elif eval_mode.startswith('image_cap'):
        model_bge = Visualized_BGE(
            model_name_bge='BAAI/bge-base-en-v1.5', 
            model_weight='/mnt/share/models/huggingface/rjc/models/BAAI/bge-visualized/Visualized_base_en_v1.5.pth'
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
    elif eval_mode.startswith('bbox_with_name'):
        results_dict = eval_bbox_with_name(benchmark_data, benchmark_root_path, model)
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
    elif eval_mode.startswith('count_num'):
        results_dict = eval_count_num(benchmark_data, benchmark_root_path, model)
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
    elif eval_mode.startswith('size_compare'):
        results_dict = eval_size_compare(benchmark_data, benchmark_root_path, model)
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
    elif eval_mode.startswith('size_choice'):
        results_dict = eval_size_choice(benchmark_data, benchmark_root_path, model)
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        del model
        torch.cuda.empty_cache()
        return results_dict
    elif eval_mode.startswith('location_choice'):
        results_dict = eval_size_choice(benchmark_data, benchmark_root_path, model)
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