# main.py
import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb

import matplotlib.pyplot as plt
transformers.logging.set_verbosity_error()

#TODO:这里是新添加的Jacobian
from utils.jacobian_calculator import JacobianCalculator  # 或你的模块路径
import yaml
from utils.visualize_jacobian import visualize_and_log_to_wandb

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=2_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=1.0)   
    parser.add_argument("--run_name", type=str, default="default")
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


def main():
    args = parse_args(None)

    with open("exp_config/conf.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config.get("jacobian", {}).get("calculation", False):
        # 训练 + Jacobian 分析
        train_full_model(args)

    if config["jacobian"].get("visualize", False):
        model_name = args.run_name
        project_dir = "results/Jacobian"
        tokens = config["jacobian"].get("tokens", [0])
        step = config["jacobian"].get("step", None)

        # 自动查找最近的 npz 文件
        if step is None:
            pattern = os.path.join(project_dir, f"{model_name}_step_*_jacobian.npz")
            matched_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if matched_files:
                latest_file = os.path.basename(matched_files[0])
                import re
                match = re.search(r"step_(\d+)_jacobian\.npz", latest_file)
                if match:
                    step = int(match.group(1))
                    print(f"✅ 未指定 step，自动使用最近文件: {latest_file} (step={step})")
                else:
                    raise FileNotFoundError(f"❌ 无法解析 step 值：{latest_file}")
            else:
                raise FileNotFoundError(f"❌ 未找到匹配文件: {pattern}")

        visualize_and_log_to_wandb(
            model_name=model_name,
            step=step,
            tokens=tokens,
            project_dir=project_dir
        )


if __name__ == "__main__":
    main()
