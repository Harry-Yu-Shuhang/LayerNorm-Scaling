#!/bin/bash

# 从参数中读取
norm_type=$1
post_num=$2

export NORM_TYPE=$norm_type
export POST_NUM=$post_num

# 从 conf.yaml 中读取 learning_rate
learning_rate=$(python -c "import yaml; print(yaml.safe_load(open('exp_config/conf.yaml'))['training']['learning_rate'])")

# 自动获取分配到的 GPU 数量（默认为1）
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

# 默认使用所有可用 GPU
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

echo "🚀 开始训练 9M 模型 | Norm: $norm_type | Post: $post_num | LR: $learning_rate | GPUs: $NUM_GPUS"
echo "🔧 使用的 GPU: $CUDA_VISIBLE_DEVICES"

# 启动训练任务
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=$NUM_GPUS --master_port=29511 main_new.py \
    --model_config configs/llama_9m.json \
    --lr $learning_rate \
    --batch_size 4 \
    --total_batch_size 32 \
    --num_training_steps 10000 \
    --warmup_steps 500 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "9m_res_${norm_type}_lr${learning_rate}" \
    --save_dir "9m_res_${norm_type}_lr${learning_rate}"
