r1_v_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main
cd ${r1_v_path}
# 睡眠10分钟，等待资源释放或系统稳定
echo "sleep 10 minutes..."
sleep 600
echo "sleep end, continue..."

# model_path=/newdisk/public/wws/00-Model-AIGC/Qwen2.5-VL-3B-Instruct
# model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/data/Qwen2.5-VL-3B-Instruct-HCD-SFT-GPT4o
model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/src/r1-v/output/qwen2_5_vl_SFT_4o_GRPO_HCD_FHM/checkpoint-1062

batch_size=4
# output_path=${r1_v_path}/src/eval/output/FHM_test/eval/res@qwen2vl-2B-checkpoint-2epoch.json
output_path=${r1_v_path}/src/eval/output/FHM_test/eval/res@qwen2.5-vl-3B-Instruct-HCD-SFT-GPT4o-GRPO-1062.json
prompt_path=${r1_v_path}/src/eval/FHM/FHM_test_seen.jsonl
gpu_ids=0,1,2,3

python src/eval/test_qwen2_5_VL_HCD_multigpu.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --gpu_ids ${gpu_ids}
