# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    # # 将完整的completion写入日志以便调试
    # if os.getenv("DEBUG_MODE") == "true":
    #     log_path = os.getenv("LOG_PATH")
    #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    #     try:
    #         with open(log_path, "a", encoding='utf-8') as f:
    #             f.write(f"------------- {current_time} 完整的completion -------------\n")
    #             f.write(f"completions: {json.dumps(completions, ensure_ascii=False, indent=2)}\n")
    #     except Exception as e:
    #         print(f"Warning: Failed to write completions to log file: {e}")
    rewards = []
    images = kwargs['image']
    image_paths = kwargs.get('image_path', ['unknown_path'] * len(images))  # 获取image_path，如果不存在则使用默认值
    # print("kwargs:")
    # print("kwargs:")
    # print("kwargs:")
    # print("kwargs: ", kwargs)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for i, (image, content, sol) in enumerate(zip(images, contents, solution)):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = next((match.strip() for match in matches if match.strip()), content.strip())
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            try:
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_paths[i]}\n")
                    f.write(f"image: {image}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
            except Exception as e:
                print(f"Warning: Failed to write to log file: {e}")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import Dataset, DatasetDict

def load_HCD_dataset(jsonl_path, image_prefix=None, num_workers=16):
    print("[Loading] HCD dataset from: ", jsonl_path)
    
    # 使用分批处理
    batch_size = 1000
    dataset = None
    
    with open(jsonl_path, 'r') as file:
        lines = file.readlines()
        total_batches = (len(lines) + batch_size - 1) // batch_size
        
        # 创建线程池用于处理批次
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            def process_batch(batch_idx):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(lines))
                batch_lines = lines[start_idx:end_idx]
                
                # 准备转换后的数据列表
                transformed_data = {
                    'image_path': [],  # 添加image_path字段
                    'image': [],
                    'problem': [],
                    'solution': []
                }
                
                # 使用线程池处理图片加载
                with ThreadPoolExecutor(max_workers=num_workers) as img_executor:
                    def process_line(line):
                        entry = json.loads(line)
                        image_path = os.path.join(image_prefix, entry['img'])
                        image = Image.open(image_path).convert('RGB')
                        text = entry['text']
                        
                        problem = f"As a harmful-content detection expert, you are presented with a sample containing an image and the overlaid text \"{text}\". Considering both the visual and textual information, what is your judgment (harmful / harmless)?"
                        
                        label = entry['label']
                        solution = "<answer> harmless </answer>" if label == 0 else "<answer> harmful </answer>"
                        
                        return (image_path, image, problem, solution)
                    
                    results = list(img_executor.map(process_line, batch_lines))
                
                # 整理结果
                for img_path, img, prob, sol in results:
                    transformed_data['image_path'].append(img_path)
                    transformed_data['image'].append(img)
                    transformed_data['problem'].append(prob)
                    transformed_data['solution'].append(sol)
                
                # 创建当前批次的dataset
                return Dataset.from_dict(transformed_data)
            
            # 并行处理所有批次
            batch_datasets = list(tqdm(
                executor.map(process_batch, range(total_batches)),
                total=total_batches,
                desc="Processing batches"
            ))
            
            # 合并所有批次的数据集
            dataset = batch_datasets[0]
            for batch_dataset in tqdm(batch_datasets[1:], desc="Merging datasets"):
                dataset = concatenate_datasets([dataset, batch_dataset])
    
    dataset_dict = DatasetDict({'train': dataset})
    print("[Finished] loaded HCD dataset from: ", jsonl_path)
    
    return dataset_dict


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    image_prefix = "/newdisk/public/wws/00-Dataset-AIGC/FHM_new/"
    dataset = load_HCD_dataset(script_args.dataset_name, image_prefix=image_prefix)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question} Only output the thinking process in <think> </think> and final judgment (harmful / harmless) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
