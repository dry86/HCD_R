import os
import json
from PIL import Image
from datasets import Dataset
from tqdm import tqdm

def convert_json_to_parquet(json_path, output_path):
    """将JSON文件转换为Parquet格式"""
    print(f"[Loading] Converting JSON file: {json_path}")
    
    # 准备数据
    transformed_data = {
        'image_path': [],
        'image': [],
        'problem': [],
        'thinking': [],
        'answer': [],
        'gt': []
    }
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每个样本
    for item in tqdm(data, desc="Processing samples"):
        # 获取图片路径并加载图片
        image_path = item['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            continue
            
        # 添加到transformed_data中
        transformed_data['image_path'].append(image_path)
        transformed_data['image'].append(image)
        transformed_data['problem'].append(item['problem'])
        transformed_data['thinking'].append(item['thinking'])
        transformed_data['answer'].append(item['answer'])
        transformed_data['gt'].append(item['gt'])
    
    # 创建dataset并保存为parquet格式
    dataset = Dataset.from_dict(transformed_data)
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存为parquet文件
    output_file = os.path.join(output_path, "merged_correct_samples-409.parquet")
    dataset.to_parquet(output_file)
    
    print(f"[Finished] Converted {len(dataset)} examples to parquet format")
    print(f"Dataset saved to {output_file}")

def main():
    # 设置输入输出路径
    json_path = "/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset_v3/merged_correct_samples-409.json"
    output_path = "/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset_v3"
    
    # 执行转换
    convert_json_to_parquet(json_path, output_path)

if __name__ == "__main__":
    main() 