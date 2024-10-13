import os
from PIL import Image
import base64
import requests
import json
import io
import time
import traceback

def compress_image(image_path, max_size_mb=10):
    max_size_bytes = max_size_mb * 1024 * 1024
    img = Image.open(image_path)
    
    # 压缩图片直到大小小于max_size_bytes
    quality = 95
    while True:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        size = buffer.tell()
        if size <= max_size_bytes or quality <= 10:
            break
        quality -= 5
    
    buffer.seek(0)
    return buffer

def encode_image_to_base64(image_path):
    image_size = os.path.getsize(image_path)
    max_size_bytes = 10 * 1024 * 1024
    
    if image_size > max_size_bytes:
        compressed_image = compress_image(image_path)
        encoded_image = base64.b64encode(compressed_image.read()).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    return encoded_image

def ask_gpt(model_name, system_prompt, user_prompt, image_list, retries=20, delay=10):
    if isinstance(image_list, str):
        image_list = [image_list]
    
    content = [
        {
            'type': 'text',
            'text': user_prompt
        }
    ]
    for image_url in image_list:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })

    url = "" # GPT API address

    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]
    })
    
    headers = {
        # 'Authorization
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # 检查请求是否成功
            return json.loads(response.text)['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt + 1 == retries:
                traceback.print_exc()
                return "error"
            time.sleep(delay)  # 等待一段时间再重试