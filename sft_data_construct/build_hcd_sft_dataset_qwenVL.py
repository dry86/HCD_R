import os
import json
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import argparse



def load_teacher_model(model_path, device):
    """加载教师模型用于生成thinking过程"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    return model, processor

def generate_thinking(model, processor, image, problem, device):
    """使用教师模型生成thinking过程"""

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final judgment (harmful / harmless) in <answer> </answer> tags."
    prompt = QUESTION_TEMPLATE.format(Question=problem)

    # 构建多模态输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # 组装 Qwen 的输入
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # 与原代码同样的解码逻辑（只截取新生成部分）
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    
    # 提取thinking部分
    try:
        thinking = response.split("<think>")[1].split("</think>")[0].strip()
    except:
        thinking = "Unable to analyze the content properly."
    
    try:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
    except:
        answer = "Unable to analyze the content properly."

    # 添加标签
    thinking = f"<think> {thinking} </think>"
    answer = f"<answer> {answer} </answer>"

    return thinking, answer

def build_hcd_dataset(jsonl_path, image_prefix, output_path, model_path, device):
    """构建HCD SFT数据集"""
    print(f"[Loading] Building HCD dataset from: {jsonl_path}")
    
    # 加载教师模型
    model, processor = load_teacher_model(model_path, device)
    
    # 准备数据
    transformed_data = {
        'image': [],
        'problem': [],
        'thinking': [],
        'answer': [],
        'solution': []
    }
    
    # 读取jsonl文件
    with open(jsonl_path, 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing dataset"):
            entry = json.loads(line)
            
            # 获取图片路径
            image_path = os.path.join(image_prefix, entry['img'])
            image = Image.open(image_path).convert('RGB')
            text = entry['text']
            
            # 构建问题
            problem = (
                f"As a harmful-content detection expert, you are presented with a sample containing an image and the overlaid text \"{text}\". "
                f"Considering both the visual and textual information, what is your judgment (harmful / harmless)?"
            )
            
            # 生成thinking过程
            thinking, answer = generate_thinking(model, processor, image, problem, device)
            
            # 构建解决方案
            label = entry['label']
            solution = "<answer> harmful </answer>" if label == 1 else "<answer> harmless </answer>"
            
            # 添加到数据集中
            transformed_data['image'].append(image)
            transformed_data['problem'].append(problem)
            transformed_data['thinking'].append(thinking)
            transformed_data['answer'].append(answer)
            transformed_data['solution'].append(solution)
    
    # 创建dataset
    dataset = Dataset.from_dict(transformed_data)
    
    # 保存为parquet格式
    os.makedirs(output_path, exist_ok=True)
    dataset.to_parquet(os.path.join(output_path, "train.parquet"))
    
    print(f"[Finished] Built HCD dataset with {len(dataset)} examples")
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Build HCD SFT dataset")
    parser.add_argument("--jsonl_path", type=str, default="/newdisk/public/wws/00-Dataset-AIGC/FHM_new/dev.jsonl", help="Path to the input jsonl file")
    parser.add_argument("--image_prefix", type=str, default="/newdisk/public/wws/00-Dataset-AIGC/FHM_new/", help="Prefix path for images")
    parser.add_argument("--output_path", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/hcd_sft_dataset", help="Output path for the dataset")
    parser.add_argument("--model_path", type=str, default="/newdisk/public/wws/00-Model-AIGC/Qwen2-VL-7B-Instruct", help="Path to the teacher model")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to run the model on")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 构建数据集
    dataset = build_hcd_dataset(
        args.jsonl_path,
        args.image_prefix,
        args.output_path,
        args.model_path,
        args.device
    )
    
    print(f"Dataset saved to {args.output_path}")

if __name__ == "__main__":
    main() 