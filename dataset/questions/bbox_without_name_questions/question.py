import json
import random

data_path = 'bench.json'

center_position_question = []

with open(data_path,'r',encoding = 'utf-8') as json_data:
    data = json.load(json_data)

for i,entity in enumerate(data):
    if i <=5000:
        question = {}
        image_info = entity['analysis_img']
        if image_info['object_count'] == 1:
            question['id'] = str(entity['unique_id'])
            question['question_id'] = entity['id']
            question['category'] = entity['base_class']
            question['image'] = entity['image']
            question['text'] = f"There is a camouflaged creature in the picture. Find it and mark its location with a BoundingBox."
            question['answer'] = entity['bbox'][0]
            center_position_question.append(question.copy())

with open('questions/bbox_without_name_questions/data/bbox_position_question.jsonl', 'w') as outfile:
    for item in center_position_question:
        json_string = json.dumps(item)
        outfile.write(json_string + '\n')
