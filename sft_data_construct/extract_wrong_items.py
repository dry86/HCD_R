import json
import os

def extract_wrong_items(input_json_path, output_jsonl_path):
    # 读取原始JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    # 打开输出文件
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        # 遍历所有结果
        for item in data['results']:
            if not item['is_correct']:
                # 提取所需字段
                output_item = {
                    "id": str(item['question']['id']),
                    "image": item['question']['image'],
                    "ground_truth": item['ground_truth'],
                    "text": item['question']['text']
                }
                # 写入JSONL文件
                f.write(json.dumps(output_item) + '\n')

if __name__ == "__main__":
    input_path = "src/eval/output/FHM_test/eval/res@qwen2vl-2B-HCD-SFT-train.json"
    output_path = "sft_data_construct/wrong_items.jsonl"
    extract_wrong_items(input_path, output_path)
    print(f"处理完成，结果已保存到: {output_path}") 