# MM-CamObj
**(AAAI25)** This is the official code repository for *"MM-CamObj: A Comprehensive Multimodal Dataset for Camouflaged Object Scenarios."*

## Dataset
- Collected by Fan *et al.*: [SINet-V2](https://github.com/GewelsJI/SINet-V2)
  - **COD10K**: [Download COD10K](https://drive.google.com/file/d/1M8-Ivd33KslvyehLK9_IUBGJ_Kf52bWG/view?usp=sharing)
- **PlantCamo**: [Download PlantCamo](https://github.com/yjybuaa/PlantCamo)
- **CPD1K**: [Download CPD1K](https://github.com/xfflyer/Camouflaged-people-detection)
- Annotated and generated by Cheng *et al.*: [SLT-Net](https://github.com/XuelianCheng/SLT-Net)
  - **MoCA-Mask**: [Download MoCA-Mask](https://drive.google.com/file/d/1FB24BGVrPOeUpmYbKZJYL5ermqUvBo_6/view?usp=sharing)

To use the dataset, download the files and place them in the `dataset/data/` directory. Then, run `dataset/data2json.py` to convert the data into JSON format.

## Evaluation
**Make sure to update your API keys and dataset paths accordingly.**
1. Modify the dataset and model paths in `MM-CamObj/camobjbench/run_eval_batch.sh` to match your local setup.
2. Execute the following command to start the evaluation:  
   ```bash
   bash MM-CamObj/camobjbench/run_eval_batch.sh
   ```
3. For evaluating with GPT Gemini, run the Python scripts located in the following directories:  
   - `MM-CamObj/camobjbench/eval_gemini-1.5-pro/questions`
   - `MM-CamObj/camobjbench/eval_GPT4o/questions`
   - `MM-CamObj/camobjbench/eval_GPT4o_mini/questions`

## Acknowledgement
Thanks to the open-source code from [Mantis](https://tiger-ai-lab.github.io/Mantis/)
