import json
import random
from pathlib import Path

def select_data(input_file, output_file, num_samples_per_class=737):
    # 读取所有数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 按label分类
    label_0_data = [item for item in data if item['label'] == 0]
    label_1_data = [item for item in data if item['label'] == 1]
    
    # 随机选择数据
    selected_label_0 = random.sample(label_0_data, num_samples_per_class)
    selected_label_1 = random.sample(label_1_data, num_samples_per_class)
    
    # 合并选中的数据
    selected_data = selected_label_0 + selected_label_1
    
    # 随机打乱数据
    random.shuffle(selected_data)
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in selected_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已选择 {num_samples_per_class} 个label=0的样本")
    print(f"已选择 {num_samples_per_class} 个label=1的样本")
    print(f"总共选择了 {len(selected_data)} 个样本")
    print(f"数据已保存到: {output_file}")

if __name__ == "__main__":
    # input_file = "/newdisk/public/wws/00-Dataset-AIGC/FHM_new/train.jsonl"
    # output_file = "selected_data.jsonl"
    # select_data(input_file, output_file)
    input_file = "/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/src/eval/FHM/train_wrong_items.jsonl"
    output_file = "/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/src/eval/FHM/selected_wrong_items_737_samples.jsonl"
    select_data(input_file, output_file)
