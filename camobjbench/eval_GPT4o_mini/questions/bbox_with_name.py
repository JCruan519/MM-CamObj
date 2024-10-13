from GPT_API import ask_gpt
from PIL import Image
import base64
from io import BytesIO
import mimetypes
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# 将本地图片转换为数据URL
def local_image_to_data_url(image_path, max_size=1024):
    try:
        with Image.open(image_path) as image:
            w, h = image.size
            scale = min(max_size / w, max_size / h)
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            base64_encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

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

# 读取问题数据
benchmark_root_path = '/cluster/home/user1/ywz/workspace/camouflaged-benchmark'
resault_path = '/cluster/home/user1/ywz/workspace/MLLMBenchPipline/camobjbench/eval_GPT4o_mini/resaults/bbox_with_name.json'
question_path = '/cluster/home/user1/ywz/workspace/camouflaged-benchmark/questions/bbox_with_name_questions/data/bbox_position_question.jsonl'

with open(question_path, 'r', encoding='utf-8') as f:
    benchmark_data = []
    for i, line in enumerate(f):
        if -1 < i < 50:
            line = line.strip()
            if line:
                data = json.loads(line)
                benchmark_data.append(data)

# 初始化计数器和列表
total_questions = len(benchmark_data)
answer_list = []
format_error = 0

# 处理单个问题
def process_question(line):
    global format_error  # 声明格式错误计数器为全局变量
    result_dict = {
        'question_id': line['question_id'],
        'image_id': int(line['id'].split("_")[0]),
        'true_answer': line['answer'],
        'model_answer': ''
    }
    true_answer = line['answer']
    question = line['text']
    image_path = line['image']
    inputs = {
        'question': question,
        'image': benchmark_root_path + '/' + image_path
    }
    prompt = '''
    You are an expert in camouflage biological monitoring. Your task is to detect camouflaged creatures in pictures and mark its location with a BoundingBox.
    **IMPORTANT**
    1.Your answer should be the BoundingBox of the camouflaged creatures
    2.Don't interpret your answer in any way
    3.If you are not sure where the camouflaged creature is, provide the bounding box where the creature is most likely to be.
    4.Your answer must be in the formart [x1, y1, x2, y2] where x and y are in the range of [0,1]
    Here is the question:\n
    '''
    question = prompt + question

    try:
        image_data_url = local_image_to_data_url(inputs['image'])
        if not image_data_url:
            raise ValueError("Failed to process image")

        model_answer = ask_gpt('gpt-4o-mini', "", question, image_data_url)
        print("#"*10)
        print(model_answer)
        model_answer = extract_bbox_from_text(model_answer)
        print(model_answer)
        print("#"*10)
    except Exception as e:
        print(f"Error processing question {line['question_id']}: {e}")
        model_answer = [0, 0, 0, 0]

    if model_answer == [0, 0, 0, 0]:
        format_error += 1

    result_dict['model_answer'] = model_answer
    return result_dict

# 使用线程池并行处理问题
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process_question, line) for line in benchmark_data]
    for future in tqdm(as_completed(futures), total=len(futures)):
        answer_list.append(future.result())

# 提取所有真实答案和模型答案
all_true_labels = [answer['true_answer'] for answer in answer_list]
all_model_answers = [answer['model_answer'] for answer in answer_list]

# 计算性能指标
miou = calculate_miou(all_true_labels, all_model_answers)

# 打印结果
print(f"Total Questions: {total_questions}")
print(f"Mean IoU: {miou:.4f}")
print(f"Format Errors: {format_error}")

# 保存结果到文件
resault = {
    'performance_metrics': {
        'miou': miou,
        'format_error': format_error
    },
    'answer_list': answer_list
}

with open(resault_path, 'w', encoding='utf-8') as f:
    json.dump(resault, f, ensure_ascii=False, indent=4)
