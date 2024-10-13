from GPT_API import ask_gpt
from PIL import Image
import base64
from io import BytesIO
import mimetypes
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# 将本地图片转换为数据URL
def local_image_to_data_url(image_path, max_size=1024):
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

benchmark_root_path = '/cluster/home/user1/ywz/workspace/camouflaged-benchmark'
resault_path = '/cluster/home/user1/ywz/workspace/MLLMBenchPipline/camobjbench/eval_GPT4o_mini/resaults/size_compare.json'
question_path = '/cluster/home/user1/ywz/workspace/camouflaged-benchmark/questions/size_compare_questions/data/size_compare_questions.jsonl'

# 读取问题数据
with open(question_path, 'r', encoding='utf-8') as f:
    benchmark_data = []
    for i, line in enumerate(f):
        if -1<i < 5000:
            line = line.strip()
            if line:
                data = json.loads(line)
                benchmark_data.append(data)

# 初始化计数器和列表
total_questions = len(benchmark_data)
answer_list = []

# 处理单个问题
def process_question(line):
    result_dict = {
        'question_id': line['question_id'],
        'image_id': int(line['id'].split("_")[0]),
        'true_answer': line['answer'],
        'model_answer': ''
    }
    true_answer = line['answer']
    question = line['text']
    image_path_list = line['image']
    inputs = {
        'question': question,
        'image_list': [benchmark_root_path + '/' + image_path for image_path in image_path_list]
    }
    try:
        model_answer = ask_gpt('gpt-4o-mini', "", inputs['question'], [local_image_to_data_url(image_path) for image_path in inputs['image_list']])
    except Exception as e:
        print(e)
        model_answer = "wrong"
    if model_answer.startswith('A'):
        model_answer = "A"
    elif model_answer.startswith('B'):
        model_answer = "B"
    elif model_answer.startswith('C'):
        model_answer = "C"
    elif model_answer.startswith('D'):
        model_answer = "D"
    if model_answer not in ['A', 'B', 'C', 'D']:
        model_answer = 'A'
    result_dict['model_answer'] = model_answer
    return result_dict

# 使用线程池并行处理问题
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_question, line) for line in benchmark_data]
    for future in tqdm(as_completed(futures), total=len(futures)):
        answer_list.append(future.result())

# 提取所有真实答案和模型答案
all_true_labels = [answer['true_answer'] for answer in answer_list]
all_model_answers = [answer['model_answer'] for answer in answer_list]

# 计算性能指标
accuracy = accuracy_score(all_true_labels, all_model_answers)
precision = precision_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
recall = recall_score(all_true_labels, all_model_answers, average='macro', zero_division=0)
f1 = f1_score(all_true_labels, all_model_answers, average='macro', zero_division=0)

# 统计正确答案和格式错误
correct = sum(1 for answer in answer_list if answer['true_answer'] == answer['model_answer'])
format_error = sum(1 for answer in answer_list if answer['model_answer'] == 'A' and answer['true_answer'] not in ['A', 'B', 'C', 'D'])

# 打印结果
print(f"Correct Questions: {correct}")
print(f"Total Questions: {total_questions}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Wrong formart answer : {format_error}")

# 保存结果到文件
resault = {
    'performance_metrics': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    },
    'answer_list': answer_list
}

with open(resault_path, 'w', encoding='utf-8') as f:
    json.dump(resault, f, ensure_ascii=False, indent=4)
