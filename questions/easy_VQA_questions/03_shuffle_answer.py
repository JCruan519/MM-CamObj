import random
import json

random.seed(41)

read_path = 'data/02_bench_question_divided.json'
write_path = 'data/03_bench_question_divided.json'
# 假设q是一个包含问题和答案的字典
with open(read_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for i, entity in enumerate(data):
        question_list = entity['question_list']
        for j, q in enumerate(question_list):
            answer = q['answer']

            # 将问题文本分割成行
            options_with_answers = q['question'].split('\n')
            question = options_with_answers[0]
            # 移除空行和空格，只留下选项行
            options = [option.lstrip() for option in options_with_answers if option.strip() != '']

            # # 找到以"answer."开头的选项的索引
            # correct_answer_index = options.index(next(option for option in options if option.startswith(f"{answer}.")))
            if answer == 'A' or answer == 'A ' or answer == 'A  ' or answer == 'A.':
                correct_answer = options[1]
            elif answer == 'B' or answer == 'B ' or answer == 'B  ' or answer == 'B.':
                correct_answer = options[2]
            elif answer == 'C' or answer == 'C ' or answer == 'C  ' or answer == 'C.':
                correct_answer = options[3]
            elif answer == 'D' or answer == 'D ' or answer == 'D  ' or answer == 'D.':
                correct_answer = options[4]
            else:
                print("Wrong 1",entity['id'],j)

            options = options[1:]

            random.shuffle(options)
            correct_option_ID = options.index(correct_answer)
            if correct_option_ID == 0:
                answer = 'A'
            elif correct_option_ID == 1:
                answer = 'B'
            elif correct_option_ID == 2:
                answer = 'C'
            elif correct_option_ID == 3:
                answer = 'D'
            else:
                print("Wrong 2",entity['ID'],j)

            Answer_list = ["A", "B", "C", "D"]
            new_options = []
            for ID, option in enumerate(options):
                option = Answer_list[ID] + ". " + option.split('.')[1]
                new_options.append(option)
            new_options = '\n'.join(new_options)
            q['question'] = question + '\n' + new_options + '\n'
            q['answer'] = answer
            data[i]['question_list'][j] = q
with open(write_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# 打印正确答案的索引
# print(correct_answer_index)  # 输出: 3
