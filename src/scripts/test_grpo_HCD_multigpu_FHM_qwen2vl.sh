r1_v_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main
cd ${r1_v_path}

# model_path=${r1_v_path}/src/r1-v/output/checkpoint-1004
# model_path=/newdisk/public/wws/00-Model-AIGC/Qwen2.5-VL-7B-Instruct
# model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/data/Qwen2-VL-2B-Instruct
# model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/src/r1-v/output/SFT_4o_GRPO_HCD_FHM/checkpoint-1062
# model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/src/r1-v/output/qwen2vl_SFT_4o_GRPO_HCD_FHM/checkpoint-500
model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/src/r1-v/output/qwen2vl_SFT_4o_GRPO_HCD_FHM_0515

batch_size=4
# output_path=${r1_v_path}/src/eval/output/FHM_test/eval/res@qwen2vl-2B-checkpoint-2epoch.json
output_path=${r1_v_path}/src/eval/output/FHM_test/eval/res@qwen2vl-2B-HCD-SFT-GRPO-wrong-items-0515-train-data.json
prompt_path=${r1_v_path}/src/eval/FHM/train_wrong_items_2.jsonl

image_prefix=/newdisk/public/wws/00-Dataset-AIGC/FHM_new    # 
gpu_ids=0,2,3

python src/eval/test_qwen2vl_HCD_multigpu.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --image_prefix ${image_prefix} \
    --gpu_ids ${gpu_ids}
