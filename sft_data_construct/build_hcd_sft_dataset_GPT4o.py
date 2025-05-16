import os
import json
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import base64
from io import BytesIO
from openai import OpenAI
import argparse

def encode_image_to_base64(image_path):
   with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode("utf-8")

def load_teacher_model(api_key):
    """设置OpenAI API密钥"""
    global client
    client = OpenAI(
        base_url="https://api.openai-proxy.org/v1",
        api_key=api_key
    )
    return None, None  # 返回None因为不需要加载本地模型

def generate_thinking(model, processor, image_path, problem, device):
    """使用GPT-4V生成thinking过程"""
    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final judgment (harmful / harmless) in <answer> </answer> tags."
    prompt = QUESTION_TEMPLATE.format(Question=problem)

    # 将图像转换为base64
    base64_image = encode_image_to_base64(image_path)

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        
        response_text = response.choices[0].message.content

        # 提取thinking部分
        try:
            thinking = response_text.split("<think>")[1].split("</think>")[0].strip()
        except:
            thinking = "Unable to analyze the content properly."
        
        try:
            answer = response_text.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            answer = "Unable to analyze the content properly."

        # 添加标签
        thinking = f"<think> {thinking} </think>"
        answer = f"<answer> {answer} </answer>"

        return thinking, answer
    except Exception as e:
        print(f"Error in generating response: {str(e)}")
        return "<think> Error in analysis </think>", "<answer> Error in analysis </answer>"

def build_hcd_dataset(jsonl_path, image_prefix, output_path, api_key, device):
    """构建HCD SFT数据集"""
    print(f"[Loading] Building HCD dataset from: {jsonl_path}")
    
    # 设置API密钥
    model, processor = load_teacher_model(api_key)
    
    # 准备数据
    transformed_data = {
        'image_path': [],
        'image': [],
        'problem': [],
        'thinking': [],
        'answer': [],
        'solution': []
    }
    
    # 读取jsonl文件
    with open(jsonl_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(tqdm(lines, desc="Processing dataset")):
            # if i >= 3:
            #     break
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
            thinking, answer = generate_thinking(model, processor, image_path, problem, device)
            
            # 构建解决方案
            label = entry['label']
            solution = "<answer> harmful </answer>" if label == 1 else "<answer> harmless </answer>"
            
            # 添加到数据集中
            transformed_data['image_path'].append(image_path)
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
    parser.add_argument("--output_path", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset", help="Output path for the dataset")
    parser.add_argument("--api_key", type=str, default="sk-WNnYtWWa5prkX9TjIE2W85jXKqrSVmEY5Nix4u345tncyoRl", help="OpenAI API key")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to run the model on (not used with GPT-4o)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 构建数据集
    dataset = build_hcd_dataset(
        args.jsonl_path,
        args.image_prefix,
        args.output_path,
        args.api_key,
        args.device
    )
    
    print(f"Dataset saved to {args.output_path}")

if __name__ == "__main__":
    main() 