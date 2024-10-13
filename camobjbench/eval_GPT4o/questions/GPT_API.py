from openai import OpenAI
import httpx
import random
import time

random.seed(41)

client = OpenAI(
    # your OpenAI API key
)

def ask_gpt(model, system_prompt, user_prompt, image_url=None, retries=1):
    for attempt in range(retries):
        try:
            if image_url:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    'type': 'text',
                                    'text': user_prompt
                                },
                                {
                                    'type': 'image_url',
                                    'image_url':{
                                        'url': image_url
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0,
                    max_tokens=400,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    'type': 'text',
                                    'text': user_prompt
                                }
                            ]
                        }
                    ],
                    temperature=0,
                    max_tokens=400,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 指数回退
            else:
                raise e

def Ask_GPT4o(system_prompt, user_prompt, image_url=False):
    return ask_gpt("gpt-4o", system_prompt, user_prompt, image_url)

def Ask_GPT_35(system_prompt, user_prompt):
    return ask_gpt("gpt-3.5-turbo-0125", system_prompt, user_prompt)

def ask_gpt_muti_image(model, system_prompt, user_prompt, image_list=None, retries=1):
    for attempt in range(retries):
        try:
            content = [
                {
                    'type': 'text',
                    'text': user_prompt
                }
            ]
            for image_url in image_list:
                content.append({'type': 'image_url','image_url':{'url': image_url}})
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0,
                max_tokens=400,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 指数回退
            else:
                raise e
# # 示例调用
# answer = Ask_GPT_35('hello world', '')
# print(answer)
