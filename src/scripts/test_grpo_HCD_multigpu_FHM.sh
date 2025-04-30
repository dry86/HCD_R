r1_v_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main
cd ${r1_v_path}

# model_path=${r1_v_path}/src/r1-v/output/checkpoint-1004
model_path=/newdisk/public/wws/00-Model-AIGC/Qwen2-VL-2B-Instruct
# model_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main/src/r1-v/output/checkpoint-1062
batch_size=4
# output_path=${r1_v_path}/src/eval/output/FHM_test/eval/res@qwen2vl-2B-checkpoint-2epoch.json
output_path=${r1_v_path}/src/eval/output/FHM_test/eval/res@qwen2vl-2B-Instruct-test.json
prompt_path=${r1_v_path}/src/eval/FHM/FHM_test_seen.jsonl
gpu_ids=0,1,2,3

python src/eval/test_qwen2vl_HCD_multigpu.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --gpu_ids ${gpu_ids}
