import json
from tqdm import tqdm

answer_path = 'data/answers_llava-v1.5-7b.jsonl'
GT_path = 'data/04_bench_question_llava.jsonl'
write_path = 'data/eval_llava-v1.5-7b.jsonl'

# 初始化4x4的计数矩阵
answer_counts = [[0 for _ in range(4)] for _ in range(4)]
answers = ['A', 'B', 'C', 'D']

# 读取模型输出和GT数据
with open(answer_path, 'r', encoding='utf-8') as f:
    outputs = [json.loads(line) for line in f]

with open(GT_path, 'r', encoding='utf-8') as f:
    GTs = [json.loads(line) for line in f]

eval_data = []

# 处理数据，填充计数矩阵
for GT in tqdm(GTs):
    for output in outputs:
        if GT['question_id'] == output['question_id']:
            GT['model_output'] = output['text']
            answer_index = answers.index(GT['answer'])
            try:
                model_answer_index = answers.index(output['text'])
                answer_counts[answer_index][model_answer_index] += 1
                eval_data.append(GT)
            except:
                print(GT['question_id'])
# 写入处理后的数据
with open(write_path, 'w', encoding='utf-8') as F:
    json.dump(eval_data, F, ensure_ascii=False, indent=4)

# 打印4x4表格的头部
print("Model Answer | A      B     C     D", end="")
for _ in answers:
    print("{:>6}".format(""), end="")
print()

# 打印分隔线
print("-" * 35)  # 根据列的数量调整横线的长度

# 打印表格的内容
for i, row in enumerate(answer_counts):
    print(f"{answers[i]}         |", end="")
    for count in row:
        print("{:>6}".format(count), end="")
    print()  # 换行

# 打印准确率等统计信息
true_count = 0
for i, row in enumerate(answer_counts):
    for j,cell in enumerate(row):
        if i == j :
            true_count+=cell


total_count = len(GTs)
print(f"Accuracy: {true_count / total_count:.2%}")
"""
行'A'，列'A'的单元格：真实答案是'A'，模型也预测为'A'的次数。
行'A'，列'B'的单元格：真实答案是'A'，但模型预测为'B'的次数。
行'B'，列'A'的单元格：真实答案是'B'，但模型预测为'A'的次数。
"""