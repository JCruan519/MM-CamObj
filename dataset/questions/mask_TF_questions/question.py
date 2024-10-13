import json
import random

prompt = "\nHere are two images. The first image is a picture containing a camouflaged creature, and the second image is a black-white mask image that outlines the camouflaged creature. Your task is to determine if the mask correctly matches the camouflaged creature in the first image. Please respond with 'True' if the mask matches the camouflaged creature, or 'False' if it does not. Your answer should be only 'True' or 'False'."

read_path="bench.json"
write_path="questions/mask_TF_questions/data/mask_FT_questions_50.jsonl"
with open(read_path, 'r', encoding="utf-8") as f:
    image_data = json.load(f)

total_data = []

for i, image in enumerate(image_data):
    if i <= 50:
        # Correct mask
        correct_mask = image['mask']

        # Incorrect mask
        mask_withouti = [entity_temp['mask'] for j, entity_temp in enumerate(image_data) if j != i]
        incorrect_mask = random.sample(mask_withouti, 1)[0]

        # Create a correct entry
        correct_data_json = {
            'id': image['unique_id'],
            'question_id': image['id'],
            'category': image['base_class'],
            'image': [image['image'], correct_mask],  # Original image and correct mask
            'text': prompt.replace('camouflaged creature',image['base_class']),
            'answer': "True"
        }
        total_data.append(correct_data_json)

        # Create an incorrect entry
        incorrect_data_json = {
            'id': image['unique_id'],
            'question_id': image['id'],
            'category': image['base_class'],
            'image': [image['image'], incorrect_mask],  # Original image and incorrect mask
            'text': prompt.replace('camouflaged creature',image['base_class']),
            'answer': "False"
        }
        total_data.append(incorrect_data_json)

with open(write_path, 'w', encoding='utf-8') as f:
    jsonl_string = '\n'.join(json.dumps(item) for item in total_data)
    f.write(jsonl_string)