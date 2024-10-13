from get_data_PlantCAMO1250 import process_PlantCAMO1250
from get_data_MoCA_Video import process_MoCA_Video
from get_data_CamouflageData import process_CamouflageData
from get_data_COD10K_v3 import process_COD10K_v3
import json

total_list = []
data_path = "data"
write_path = 'images.json'
total_list = total_list+process_COD10K_v3(data_path+"/COD10K-v3") + process_PlantCAMO1250(data_path+"/PlantCAMO1250")+process_MoCA_Video(data_path+"/MoCA_Video")+process_CamouflageData(data_path+"/CamouflageData")
print(len(total_list))
for i,data in enumerate(total_list):
    data['id'] = i
    data['unique_id'] ="{:06d}_{}".format(data['id'], data['base_class'])
with open(write_path, 'w',encoding='utf-8') as f:
    json.dump(total_list, f,ensure_ascii=False, indent=4)