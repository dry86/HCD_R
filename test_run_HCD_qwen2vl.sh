cd src/r1-v

DATA_PATH=/newdisk/public/wws/00-Dataset-AIGC/FHM_new/train.jsonl

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_qwen2vl_SFT_4o_GRPO_HCD_FHM_20250511.txt"

# 设置环境变量以优化内存使用
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/HCD_grpo.py \
    --output_dir ./output/SFT_4o_GRPO_HCD_FHM \
    --model_name_or_path /newdisk/public/wws/01-AIGC-GPRO/R1-V-main/data/Qwen2-VL-2B-Instruct-HCD-SFT-GPT4o \
    --dataset_name ${DATA_PATH} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 256 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 200704 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-SFT-GRPO-HCD-FHM-train \
    --save_steps 500 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
