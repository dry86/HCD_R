import os
import json
import pandas as pd
from tqdm import tqdm

def parquet_to_json(parquet_path):
    # 获取输出路径（与parquet文件在同一目录）
    output_dir = os.path.dirname(parquet_path)
    output_file = os.path.join(output_dir, "train_gt.json")
    
    print(f"正在加载parquet文件: {parquet_path}")
    # 使用pandas读取parquet文件
    df = pd.read_parquet(parquet_path)
    
    print(f"开始转换为JSON格式...")
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')  # 开始JSON数组
        
        # 遍历数据集并写入JSON
        for i, row in enumerate(tqdm(df.itertuples(), total=len(df))):
            # 将每个样本转换为字典
            sample_dict = {
                'image_path': row.image_path,
                'problem': row.problem,
                'thinking': row.thinking,
                'answer': row.answer,
                'solution': row.solution
            }
            
            # 写入JSON，除了最后一个样本外都添加逗号
            if i < len(df) - 1:
                f.write(json.dumps(sample_dict, ensure_ascii=False, indent=2) + ',\n')
            else:
                f.write(json.dumps(sample_dict, ensure_ascii=False, indent=2) + '\n')
        
        f.write(']')  # 结束JSON数组
    
    print(f"转换完成！")
    print(f"JSON文件已保存到: {output_file}")
    print(f"总共转换了 {len(df)} 条数据")

if __name__ == "__main__":
    parquet_path = "/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset/train.parquet"
    parquet_to_json(parquet_path)
