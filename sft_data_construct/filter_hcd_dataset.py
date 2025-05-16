import os
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

def filter_dataset(input_path, output_path):
    """过滤数据集，只保留answer与solution匹配的数据"""
    print(f"[Loading] Loading dataset from: {input_path}")
    
    # 读取parquet文件
    df = pd.read_parquet(input_path)
    print(f"Original dataset size: {len(df)}")
    
    # 过滤条件：
    # 1. answer不等于solution的数据
    # 2. 包含"Error in analysis"的数据
    filtered_df = df[
        (df['answer'] == df['solution']) & 
        (~df['thinking'].str.contains('Error in analysis', na=False))
    ]
    
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Removed {len(df) - len(filtered_df)} samples")
    
    # 创建新的dataset
    dataset = Dataset.from_pandas(filtered_df)
    
    # 保存为新的parquet文件
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "train_gt.parquet")
    dataset.to_parquet(output_file)
    
    print(f"[Finished] Filtered dataset saved to: {output_file}")
    
    # 打印一些统计信息
    print("\nDataset Statistics:")
    print(f"Total samples: {len(filtered_df)}")
    print(f"Number of harmful samples: {len(filtered_df[filtered_df['solution'] == '<answer> harmful </answer>'])}")
    print(f"Number of harmless samples: {len(filtered_df[filtered_df['solution'] == '<answer> harmless </answer>'])}")

def main():
    # 设置输入输出路径
    input_path = "/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset/train.parquet"
    output_path = "/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset"
    
    # 过滤数据集
    filter_dataset(input_path, output_path)

if __name__ == "__main__":
    main() 