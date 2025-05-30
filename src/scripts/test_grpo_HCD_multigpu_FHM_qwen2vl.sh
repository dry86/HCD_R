r1_v_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main
cd ${r1_v_path}

# model_path=${r1_v_path}/src/r1-v/output/checkpoint-1004
# model_path=/newdisk/public/wws/00-Model-AIGC/Qwen2.5-VL-7B-Instruct
# model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/data/Qwen2-VL-2B-Instruct
# model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/src/r1-v/output/SFT_4o_GRPO_HCD_FHM/checkpoint-1062
# model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/data/Qwen2-VL-2B-Instruct-HCD-SFT-GPT4o-v2
model_path=/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/output/qwen2vl_7b/lora_sft_4e_lr1e-4

# output_path=${r1_v_path}/src/eval/output/FHM_test/eval/res@qwen2vl-2B-checkpoint-2epoch.json
output_path=${r1_v_path}/src/eval/output/FHM_test/eval/res@qwen2vl-7B-HCD-Lora-SFT-FHM-4e_lr1e-4.json
prompt_path=${r1_v_path}/src/eval/FHM/FHM_test_seen.jsonl

# image_prefix=/newdisk/public/wws/00-Dataset-AIGC/FHM_new    # 
batch_size=4
gpu_ids=3

python src/eval/test_qwen2vl_HCD_multigpu_hateful.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --gpu_ids ${gpu_ids}
