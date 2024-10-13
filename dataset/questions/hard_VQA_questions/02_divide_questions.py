import json
import traceback
import re


def divide_questions(data_in):
    # 将JSON字符串解析成Python字典
    data_dict = data_in
    # 将问题和答案从'questions'字段中提取出来，并整理到'question_list'中
    if "questions" not in data_dict:
        print(data_dict)
    questions_str = data_dict['questions']
    # questions = questions_str.split('\n\n','\n    \n')  # 假设每个问题和答案是通过两个换行符分隔的
    # 使用正则表达式来分割字符串
    questions = re.split(r'\n\s*\n', questions_str)
    data_dict['question_list'] = []
    for question in questions:
        # print(question)
        # 将问题和答案分割开
        question_lines = question.strip().split('\n')
        answer = question_lines[0].split('Answer: ')[1]  # 答案
        question_lines[0] = question_lines[0].split(' Answer: ')[0]
        question_text = "\n".join(question_lines)  # 问题文本

        # 构建问题字典
        question_dict = {
            'question': question_text,
            'answer': answer
        }

        # 将问题字典添加到列表中
        data_dict['question_list'].append(question_dict)

    return data_dict


read_path = 'data/01_bench_question.json'
write_path = 'data/02_bench_question_divided.json'

with open(read_path, 'r', encoding='utf-8') as f:
    data_questions = json.load(f)

for list_ID,data_dict in enumerate(data_questions):
    ID = data_questions[list_ID]['id']
    try:
        data_questions[list_ID] = divide_questions(data_dict)

    except Exception as e:
        print(f"样本{ID}格式错误，请手动修改")
        traceback.print_exc()
with open(write_path, 'w', encoding='utf-8') as f:
    json.dump(data_questions, f, ensure_ascii=False, indent=4)
