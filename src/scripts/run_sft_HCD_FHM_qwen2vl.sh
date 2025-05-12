r1_v_path=/newdisk/public/wws/01-AIGC-GPRO/R1-V-main
cd ${r1_v_path}

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=eno2
# export NCCL_IB_GID_INDEX=3

ACCELERATE_LOG_LEVEL=info accelerate launch --multi_gpu --num_processes 4 --config_file src/r1-v/configs/zero2.yaml src/r1-v/src/open_r1/HCD_sft.py --config src/r1-v/configs/HCD_qwen2vl_sft_config.yaml 
