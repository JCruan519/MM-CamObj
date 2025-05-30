# MM-CamObj
**(AAAI 2025)** This is the official code repository for *"MM-CamObj: A Comprehensive Multimodal Dataset for Camouflaged Object Scenarios."*

## 📊 Dataset

### Download
Download the dataset images from [Hugging Face](https://huggingface.co/datasets/Winston-Yuan/MM-CamObj)

### Setup
1. Unzip the image files into the `dataset/data/` directory
2. The dataset structure includes:
   - `dataset/questions/` - Benchmark evaluation data
   - `dataset/train_data/` - Training data with text prompts

## 🚀 Evaluation

**Prerequisites:** Ensure your API keys and dataset paths are properly configured.

### Quick Start
1. Update the dataset and model paths in `MM-CamObj/camobjbench/run_eval_batch.sh` to match your local environment
2. Run the evaluation script:
   ```bash
   bash MM-CamObj/camobjbench/run_eval_batch.sh
   ```

## 🙏 Acknowledgement
We thank the open-source community, particularly the [Mantis](https://tiger-ai-lab.github.io/Mantis/) project for their valuable contributions.
