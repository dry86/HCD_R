r1_v_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main
cd ${r1_v_path}

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file src/r1-v/configs/zero2.yaml src/r1-v/src/open_r1/sft.py --config src/r1-v/configs/qwen2vl_sft_config.yaml 
