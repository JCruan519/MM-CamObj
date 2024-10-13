import json
import random

# prompt= '''
# Here are five images. The first image is a picture containing a camouflaged creature, and the following four images are mask images that outline the camouflaged creature. Please select the mask image that best matches the camouflaged creature in the first image. The four mask images are labeled as ['A', 'B', 'C', 'D']. You only need to respond with the label of the image that is the correct match.

# *IMPORTANT*
# 1.Your answer should be just one letter in [A, B, C, D]
# 2.Don't interpret your answer in any way
# '''

prompt = "\nHere are three images. The first image is a picture containing a camouflaged creature, and the following two images are black-white mask images that outline the camouflaged creature labeled as ['A', 'B']. Please select the mask image that best matches the camouflaged creature in the first image.  You need to respond with the label of the image that is the correct match and do not give any explanations\n"

# prompt = '''
# What are in the images?
# '''

# read_path="../bench.json"
# write_path="data/mask_matching_questions.jsonl"

read_path="bench.json"
write_path="questions/mask_mach_questions/data/mask_matching_questions_50.jsonl"
total_data = []
with open(read_path,'r',encoding="utf-8") as f:
    image_data = json.load(f)
for i,image in enumerate(image_data):
    if i<=50:
        mask_withouti = [entity_temp['mask'] for j,entity_temp in enumerate(image_data) if j != i]
        random_mask = random.sample(mask_withouti,1)
        # answer_list = [0,1,2,3]
        answer_list = [0,1]
        random.shuffle(answer_list)
        choices_list = [image['mask']]+random_mask
        choices_list_new = [choices_list[num] for num in answer_list]
        if answer_list[0] == 0:
            answer = "A"
        elif answer_list[1] == 0:
            answer = "B"
        elif answer_list[2] == 0:
            answer = "C"
        elif answer_list[3] == 0:
            answer = "D"

        data_json = {
            'id':image['unique_id'],
            'question_id':image['id'],
            'catgory':image['base_class'],
            'image':[image['image']] + choices_list_new,#共5张图
            'text':prompt.replace('camouflaged creature',image['base_class']),
            'answer':answer
        }
        total_data.append(data_json)
with open(write_path,'w',encoding='utf-8') as f:
    jsonl_string = '\n'.join(json.dumps(item) for item in total_data)
    # 将字符串写入文件
    f.write(jsonl_string)