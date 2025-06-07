cd src/r1-v

DATA_PATH=/newdisk/public/wws/00-Dataset-AIGC/FHM_new/train.jsonl
# DATA_PATH=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/src/eval/FHM/selected_wrong_items_737_samples.jsonl

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_qwen2vl_SFT_4o_GRPO_HCD_FHM_20250605.txt"

# 设置环境变量以优化内存使用
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/HCD_grpo_v2.py \
    --output_dir ./output/qwen2vl_SFT_4o_GRPO_HCD_FHM_0605 \
    --model_name_or_path /newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/output/Qwen2-VL-qwen2vl_lora_sft_3e \
    --dataset_name ${DATA_PATH} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 200704 \
    --num_train_epochs 1 \
    --run_name qwen2vl-2B-SFT-GRPO-HCD-FHM-0605 \
    --save_steps 5000 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
