# Define the list of model names
model_name_list=(
    'llava' # 0
    'llava' # 1
    'qwenVL' # 2
    'minicpm-V-2.5' # 3
    'instructblip' # 5
    'fuyu' # 6
    'kosmos2' # 7
    'idefics1'
)

# Define the list of model paths
model_path_list=(
    '/mnt/share/models/huggingface/rjc/models/llava-hf/llava-1.5-7b-hf'
    '/mnt/share/models/huggingface/rjc/models/llava-hf/llava-1.5-13b-hf'
    '/mnt/share/models/huggingface/rjc/models/Qwen/Qwen-VL-Chat'
    '/mnt/share/models/huggingface/rjc/models/openbmb/MiniCPM-Llama3-V-2_5'
    '/mnt/share/models/huggingface/rjc/models/Salesforce/instructblip-flan-t5-xxl'
    '/mnt/share/models/huggingface/rjc/models/adept/fuyu-8b'
    '/mnt/share/models/huggingface/rjc/models/microsoft/kosmos-2-patch14-224'
    '/mnt/share/models/huggingface/rjc/models/HuggingFaceM4/idefics-9b-instruct'
)

# Define the list of evaluation modes
eval_mode_list=(
    'easy_VQA' 
    'hard_VQA'
    'image_cap'
    'count_choice'
    'mask_FT'
    'mask_match'
    'bbox_without_name'
)

# Define paths for datasets and image directory
easy_vqa_dataset_path='/mnt/share/models/huggingface/rjc/camobj_bench/benchmark/questions/easy_VQA_questions/data/easy_vqa.jsonl'
hard_vqa_dataset_path='/mnt/share/models/huggingface/rjc/camobj_bench/benchmark/questions/hard_VQA_questions/data/hard_vqa.jsonl'
img_cap_dataset_path='/mnt/share/models/huggingface/rjc/camobj_bench/benchmark/camobj_bench_image_caption.json'
count_choice_dataset_path='/mnt/share/models/huggingface/rjc/camobj_bench/benchmark/questions/count_choice/data/count_choice.jsonl'
mask_ft_dataset_path='/mnt/share/models/huggingface/rjc/camobj_bench/benchmark/questions/mask_FT/data/mask_ft.jsonl'
mask_match_dataset_path='/mnt/share/models/huggingface/rjc/camobj_bench/benchmark/questions/mask_match/data/mask_match.jsonl'
bbox_without_name_dataset_path='/mnt/share/models/huggingface/rjc/camobj_bench/benchmark/questions/bbox_without_name/data/bbox_without_name.jsonl'
img_path='/mnt/share/models/huggingface/rjc/camobj_bench'

# Loop through the eval modes
for eval_mode in "${eval_mode_list[@]}"; do
    # Set dataset path based on evaluation mode
    if [ "$eval_mode" == "easy_VQA" ]; then
        dataset_path=$easy_vqa_dataset_path
    elif [ "$eval_mode" == "hard_VQA" ]; then
        dataset_path=$hard_vqa_dataset_path
    elif [ "$eval_mode" == "image_cap" ]; then
        dataset_path=$img_cap_dataset_path
    elif [ "$eval_mode" == "count_choice" ]; then
        dataset_path=$count_choice_dataset_path
    elif [ "$eval_mode" == "mask_FT" ]; then
        dataset_path=$mask_ft_dataset_path
    elif [ "$eval_mode" == "mask_match" ]; then
        dataset_path=$mask_match_dataset_path
    elif [ "$eval_mode" == "bbox_without_name" ]; then
        dataset_path=$bbox_without_name_dataset_path
    fi

    # Loop through each model number
    for i in {0..7}; do
        # Set the CUDA visible devices (assuming each process uses a separate GPU)
        export CUDA_VISIBLE_DEVICES=$i

        # Run the Python script in the background for each model
        python eval_mllm.py \
            --model_name "${model_name_list[$i]}" \
            --model_path "${model_path_list[$i]}" \
            --dataset_path ${dataset_path} \
            --img_path ${img_path} \
            --results_dir /mnt/share/models/huggingface/rjc/camobj_bench/eval_res \
            --eval_mode ${eval_mode} &
    done

    # Wait for all background processes to finish before moving to the next eval_mode
    wait
done
