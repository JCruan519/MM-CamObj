import json
import random

data_path = 'bench.json'

with open(data_path,'r',encoding='utf-8') as json_data:
    data = json.load(json_data)

def generate_object_count_question(base_class,image_info):
    object_count = image_info['object_count']
    if object_count < 5:
        options = list(range(1, 5))
    else:
        options = list(range(object_count - 2, object_count + 2))

    random.shuffle(options)
    correct_index = options.index(object_count)

    question = f"How many {base_class} are in the image?\n"
    options_str = '\n'.join([f"{'ABCD'[i]}. {option}" for i, option in enumerate(options)])
    answer = 'ABCD'[correct_index]

    return question + options_str, answer

object_count_question = []

for i, entity in enumerate(data):
    if i<=5000:
        question = {}
        image_info = entity['analysis_img']
        if 0<image_info['object_count'] <= 10:
            question['id'] = str(entity['unique_id'])
            question['question_id'] = entity['id']
            question['category'] = entity['base_class']
            question['image'] = entity['image']
            question['text'], question['answer'] = generate_object_count_question(entity['base_class'],image_info)
            object_count_question.append(question.copy())

with open('questions/count_choice_questions/data/object_count_question.jsonl', 'w') as outfile:
    for item in object_count_question:
        json_string = json.dumps(item)
        outfile.write(json_string + '\n')
