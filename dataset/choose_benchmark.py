import json
import random
random.seed(41)
with  open('images.json','r',encoding='utf-8') as f:
    total_data = json.load(f)
    random.shuffle(total_data)
    bench_data = []
    for i in range(600) :
        bench_data.append(total_data[i])
with open('bench.json','w',encoding='utf-8') as f:
    json.dump(bench_data,f,ensure_ascii=False,indent=4)