import os
import json
from tqdm import tqdm
import argparse

def load_dataset(json_path):
    """加载数据集"""
    print(f"正在加载数据集: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_correct_prediction(sample):
    """检查预测是否正确"""
    predicted_answer = sample['answer'].strip().lower()
    gt_answer = sample['gt'].strip().lower()
    return predicted_answer == gt_answer

def merge_correct_samples(v1_path, v2_path, v3_path, output_path):
    """合并三个版本中预测正确的样本"""
    # 加载三个版本的数据集
    v1_data = load_dataset(v1_path)
    v2_data = load_dataset(v2_path)
    v3_data = load_dataset(v3_path)
    
    # 用于存储合并后的数据
    merged_data = []
    processed_samples = set()  # 用于记录已处理的样本
    
    # 统计信息
    stats = {
        'v1_correct': 0,
        'v2_correct': 0,
        'v3_correct': 0,
        'total_merged': 0
    }
    
    print("\n开始处理数据集...")
    
    # 处理v1数据
    for sample in tqdm(v1_data, desc="处理v1数据"):
        if is_correct_prediction(sample):
            sample_id = sample['image_path']  # 使用图片路径作为唯一标识
            if sample_id not in processed_samples:
                merged_data.append(sample)
                processed_samples.add(sample_id)
                stats['v1_correct'] += 1
    
    # 处理v2数据
    for sample in tqdm(v2_data, desc="处理v2数据"):
        if is_correct_prediction(sample):
            sample_id = sample['image_path']
            if sample_id not in processed_samples:
                merged_data.append(sample)
                processed_samples.add(sample_id)
                stats['v2_correct'] += 1
    
    # 处理v3数据
    for sample in tqdm(v3_data, desc="处理v3数据"):
        if is_correct_prediction(sample):
            sample_id = sample['image_path']
            if sample_id not in processed_samples:
                merged_data.append(sample)
                processed_samples.add(sample_id)
                stats['v3_correct'] += 1
    
    # 更新总合并数
    stats['total_merged'] = len(merged_data)
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存合并后的数据集
    output_file = os.path.join(output_path, "merged_correct_samples.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n[统计信息]")
    print(f"v1中正确的样本数: {stats['v1_correct']}")
    print(f"v2中正确的样本数: {stats['v2_correct']}")
    print(f"v3中正确的样本数: {stats['v3_correct']}")
    print(f"合并后的总样本数: {stats['total_merged']}")
    print(f"\n合并后的数据集已保存到: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="合并三个版本中预测正确的样本")
    parser.add_argument("--v1_path", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset_v3/train_v3.json", help="v1版本数据集路径")
    parser.add_argument("--v2_path", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset_v3/train_v3_reprocessed/reprocessed_errors.json", help="v2版本数据集路径")
    parser.add_argument("--v3_path", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset_v3/train_v3_reprocessed/reprocessed_errors_v2/reprocessed_errors.json", help="v3版本数据集路径")
    parser.add_argument("--output_path", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset_v3/", help="输出目录路径")
    return parser.parse_args()

def main():
    args = parse_args()
    merge_correct_samples(
        args.v1_path,
        args.v2_path,
        args.v3_path,
        args.output_path
    )

if __name__ == "__main__":
    main() 