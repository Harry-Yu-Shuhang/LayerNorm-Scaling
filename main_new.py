# main.py

import yaml
import argparse
from utils.train_full_model import train_full_model
from utils.jacobian_calculator import JacobianCalculator
from utils.visualize_jacobian import visualize_and_log_to_wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--total_batch_size", type=int)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--single_gpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    with open("exp_config/conf.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config.get("jacobian", {}).get("calculation", False):
        # 训练 + Jacobian 分析
        train_full_model(args)

    if config.get("jacobian", {}).get("visualize", False):
        # 可视化已有 .npz
        visualize_and_log_to_wandb(
            model_name=args.run_name,
            step=config["jacobian"].get("step", 0),
            tokens=config["jacobian"].get("tokens", [0]),
            project_dir="results/Jacobian"
        )

if __name__ == "__main__":
    main()
