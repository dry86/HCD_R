import os
import json
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
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
    return None, None

def generate_thinking(model, processor, image_path, problem, device):
    """使用GPT-4o生成thinking过程"""
    base64_image = encode_image_to_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": problem
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

        try:
            thinking = response_text.split("<think>")[1].split("</think>")[0].strip()
        except:
            thinking = "Unable to analyze the content properly."
        
        try:
            answer = response_text.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            answer = "Unable to analyze the content properly."

        thinking = f"<think> {thinking} </think>"
        answer = f"<answer> {answer} </answer>"

        return thinking, answer
    except Exception as e:
        print(f"Error in generating response: {str(e)}")
        return "<think> Error in analysis </think>", "<answer> Error in analysis </answer>"

def reprocess_error_samples(input_json, output_path, api_key, device):
    """重新处理预测错误的样本"""
    print(f"[Loading] 正在从 {input_json} 加载数据...")
    
    # 设置API密钥
    model, processor = load_teacher_model(api_key)
    
    # 读取原始数据集
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 准备数据
    transformed_data = {
        'image_path': [],
        'image': [],
        'problem': [],
        'thinking': [],
        'answer': [],
        'gt': [],
        'original_answer': []  # 保存原始预测结果
    }
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 创建json文件
    json_output_path = os.path.join(output_path, "reprocessed_errors.json")
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json_file.write('[\n')
    
    error_count = 0
    reprocessed_count = 0
    
    # 处理每个样本
    for i, entry in enumerate(tqdm(dataset, desc="正在重新处理错误样本")):
        # 检查预测是否正确
        predicted_answer = entry['answer'].strip().lower()
        gt_answer = entry['gt'].strip().lower()
        
        if predicted_answer != gt_answer:
            error_count += 1
            # 获取图片路径
            image_path = entry['image_path']
            image = Image.open(image_path).convert('RGB')
            problem = entry['problem']
            
            # 重新生成thinking过程
            thinking, answer = generate_thinking(model, processor, image_path, problem, device)
            
            # 检查新的预测是否正确
            new_predicted_answer = answer.strip().lower()
            if new_predicted_answer == gt_answer:
                reprocessed_count += 1
            
            # 构建当前样本的数据
            current_sample = {
                'image_path': image_path,
                'problem': problem,
                'thinking': thinking,
                'answer': answer,
                'gt': entry['gt'],
                'original_answer': entry['answer']  # 保存原始预测结果
            }
            
            # 添加到transformed_data中
            transformed_data['image_path'].append(image_path)
            transformed_data['image'].append(image)
            transformed_data['problem'].append(problem)
            transformed_data['thinking'].append(thinking)
            transformed_data['answer'].append(answer)
            transformed_data['gt'].append(entry['gt'])
            transformed_data['original_answer'].append(entry['answer'])
            
            # 写入json文件
            with open(json_output_path, 'a', encoding='utf-8') as json_file:
                if i > 0:
                    json_file.write(',\n')
                json.dump(current_sample, json_file, ensure_ascii=False, indent=2)
    
    # 完成json文件
    with open(json_output_path, 'a', encoding='utf-8') as json_file:
        json_file.write('\n]')
    
    # 创建dataset并保存为parquet格式
    dataset = Dataset.from_dict(transformed_data)
    dataset.to_parquet(os.path.join(output_path, "reprocessed_errors.parquet"))
    
    print(f"\n[统计信息]")
    print(f"总错误样本数: {error_count}")
    print(f"重新处理后正确的样本数: {reprocessed_count}")
    print(f"正确率提升: {reprocessed_count/error_count*100:.2f}%")
    print(f"\n[完成] 重新处理的数据已保存到 {output_path}")
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="重新处理预测错误的样本")
    parser.add_argument("--input_json", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset_v3/train_v3_reprocessed/reprocessed_errors.json", help="输入JSON文件路径")
    parser.add_argument("--output_path", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset_v3/train_v3_reprocessed/reprocessed_errors_v2", help="输出目录路径")
    parser.add_argument("--api_key", type=str, default="sk-WNnYtWWa5prkX9TjIE2W85jXKqrSVmEY5Nix4u345tncyoRl", help="OpenAI API密钥")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备（对于GPT-4o未使用）")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 重新处理错误样本
    dataset = reprocess_error_samples(
        args.input_json,
        args.output_path,
        args.api_key,
        args.device
    )

if __name__ == "__main__":
    main() 