import json
import os
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import re

def extract_answer(text):
    # 使用正则表达式提取<answer>标签中的内容
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text)
    if match:
        return match.group(1).strip().lower()
    return None

def calculate_metrics(y_true, y_pred):
    # 将标签转换为二进制形式
    mapping = {'hateful': 1, 'not-hateful': 0}
    
    # 过滤掉None值，并记录错误案例
    valid_indices = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if mapping.get(true) is not None and mapping.get(pred) is not None:
            valid_indices.append(i)
    
    # 只使用有效的数据计算指标
    y_true_bin = [mapping.get(y_true[i]) for i in valid_indices]
    y_pred_bin = [mapping.get(y_pred[i]) for i in valid_indices]
    
    # 计算错误案例数量（包括None值的情况）
    error_count = len(y_true) - len(valid_indices)
    
    # 计算各项指标
    acc = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin)
    recall = recall_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)
    
    # 计算macro F1
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro')
    
    # 计算AUROC
    try:
        auroc = roc_auc_score(y_true_bin, y_pred_bin)
    except ValueError:
        auroc = None  # 如果只有一个类别，AUROC无法计算
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1_macro,
        'auroc': auroc,
        'none_value_errors': error_count  # 添加None值错误计数
    }

def main(json_path):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取真实标签和预测标签
    y_true = []
    y_pred = []
    error_cases = []
    
    # 创建错误案例的图片目录
    error_images_dir = os.path.join(os.path.dirname(json_path), 'error_cases_images')
    os.makedirs(error_images_dir, exist_ok=True)
    
    for item in data:
        true_label = extract_answer(item['gt'])
        pred_label = extract_answer(item['answer'])
        
        # 添加所有标签用于计算指标
        if true_label:
            y_true.append(true_label)
            y_pred.append(pred_label if pred_label else '')  # 如果预测标签为空，添加空字符串
        
        # 如果预测标签与真实标签不同，保存为错例
        if true_label != pred_label:
            error_case = {
                'image_path': item['image_path'],
                'problem': item['problem'],
                'thinking': item['thinking'],
                'answer': item['answer'],
                'gt': item['gt']
            }
            error_cases.append(error_case)
            
            # 复制错误案例的图片
            image_filename = os.path.basename(item['image_path'])
            src_path = item['image_path']
            dst_path = os.path.join(error_images_dir, image_filename)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying image {src_path}: {str(e)}")
    
    # 计算评估指标
    metrics = calculate_metrics(y_true, y_pred)
    
    # 按图片文件名排序错例
    error_cases.sort(key=lambda x: os.path.basename(x['image_path']))
    
    # 保存错误案例到JSON文件
    with open(os.path.join(os.path.dirname(json_path), 'error_cases.json'), 'w') as f:
        json.dump(error_cases, f, indent=2)
    
    # 打印评估指标
    print("\n评估指标:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    if metrics['auroc'] is not None:
        print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"\nNone值错误数量: {metrics['none_value_errors']}")
    print(f"预测错误数量: {len(error_cases)}")
    print(f"总错误数量: {metrics['none_value_errors'] + len(error_cases)}")
    print(f"错误案例已保存到 error_cases.json")
    print(f"错误案例图片已保存到 {error_images_dir} 目录")

if __name__ == "__main__":
    json_path = '/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/sft_data_construct/gpt4o_hcd_sft_dataset_v3/train_v3_reprocessed/reprocessed_errors_v2/reprocessed_errors.json'
    main(json_path) 