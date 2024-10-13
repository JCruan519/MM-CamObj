import json
from prompt import choice_question2llava_prompt

read_path = 'data/03_bench_question_divided.json'
write_path = 'data/04_bench_question_llava_50.jsonl'

total_questions = []

with open(read_path, 'r',encoding='utf-8') as f:
    j = 0
    data = json.load(f)
    for p,picture in enumerate(data):
        if p < 50:
            question_list = picture['question_list']
            for i , question in enumerate(question_list):
                j+=1
                question_data = {'id': str(picture['id']) + "_" + str(i),
                                'question_id': j,
                                'category': picture['base_class'],
                                'image': picture['image'],
                                'text': choice_question2llava_prompt(question['question']),
                                'answer': question['answer']}
                total_questions.append(question_data)
# 打开一个文件用于写入
with open(write_path, 'w', encoding='utf-8') as file:
    for item in total_questions:
        # 序列化JSON对象为字符串
        json_string = json.dumps(item)
        # 写入文件，并添加换行符
        file.write(json_string + '\n')