import base64
from mimetypes import guess_type
from GPT_API import Ask_GPT_35, Ask_GPT4o, ask_gpt
from prompt import easy_choice_question,hard_choice_question
import concurrent.futures
from tqdm import tqdm
import json
import os


from PIL import Image
import base64
from io import BytesIO
import mimetypes

def local_image_to_data_url(image_path, max_size=1024):  # max_size 是图片的最大允许尺寸
    # 使用Pillow库打开图片文件
    with Image.open(image_path) as image:
        # 获取图片的原始尺寸
        w, h = image.size
        # 计算缩放比例
        scale = min(max_size / w, max_size / h)

        # 如果缩放比例小于1，说明图片尺寸超出了最大尺寸，需要缩放
        if scale < 1:
            # 计算新的尺寸
            new_w, new_h = int(w * scale), int(h * scale)
            # 等比例缩放图片
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 根据文件名猜测图片的MIME类型
        mime_type, _ = mimetypes.guess_type(image_path)
        # 如果无法猜测MIME类型，则默认为'application/octet-stream'
        if mime_type is None:
            mime_type = 'application/octet-stream'

        # 将图片保存到BytesIO对象中，这里使用PNG格式，可以根据需要更改格式
        buffer = BytesIO()
        image.save(buffer, format='PNG')

        # 将图片的二进制数据编码为base64格式
        base64_encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 构建并返回包含图片数据的URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

# 示例使用方式：
# data_url = local_image_to_data_url('图片路径.jpg', max_size=800)
# print(data_url)


def get_questions(name, image_path):
    # system_prompt, user_prompt = easy_choice_question(name)
    system_prompt, user_prompt = hard_choice_question(name)
    questions = ask_gpt('gpt-4o', system_prompt, user_prompt, local_image_to_data_url(image_path))
    return questions


def process_data(input_data):
    entry = input_data
    name = entry['base_class']
    image_path = os.path.join("../../",entry['image'])
    entry['questions'] = get_questions(name, image_path)
    with open(write_path, 'w',encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    return entry


if __name__ == '__main__':
    # An example of dataset
    read_path = '../../bench.json'
    write_path = 'data/01_bench_question.json'
    with open(read_path,'r',encoding='utf-8') as f:
        data_list = json.load(f)
    test_data_limit = 6000 # 测试时只处理前50条数据
    data_list = data_list[:test_data_limit]

    # Use ThreadPoolExecutor for parallel processing and tqdm for progress bar
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_data, entry) for entry in data_list]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing data"):
            # Ensure any exceptions are raised
            try:
                future.result()  # 这会等待任务完成，并抛出任何异常
            except Exception as e:
                print(f"An error occurred: {e}")
    # # Now data will have the 'questions' field filled
    # with open(write_path, 'w',encoding='utf-8') as f:
    #     json.dump(data_list, f, ensure_ascii=False, indent=4)
