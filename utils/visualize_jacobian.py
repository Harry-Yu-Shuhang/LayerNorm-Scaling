import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def visualize_and_log_to_wandb(model_name, step, tokens, project_dir="results/Jacobian"):
    if int(os.environ.get("RANK", 0)) != 0:
        return  # ⛔ 避免多进程重复 log

    if wandb.run is None:  # ✅ 安全初始化
        wandb.init(project="cod", name=model_name)

    file_path = os.path.join(project_dir, f"{model_name}_step_{step}_jacobian.npz")
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    jacobian = data["jacobian"].item()
    frobenius = data["frobenius"].item()
    mse = data.get("mse", None)
    if mse is not None:
        mse = mse.item()

    for token_idx in tokens:
        # === Frobenius 和 MSE 折线图，可视化成两行两列（attention + ffn）
        for metric_name in ["frobenius", "mse"]:
            plt.figure(figsize=(12, 4))
            for i, module_type in enumerate(["attention", "ffn"]):
                layers = []
                values = []
                for layer in sorted(frobenius.keys()):
                    if metric_name == "frobenius" and token_idx in frobenius[layer][module_type]:
                        layers.append(layer)
                        values.append(frobenius[layer][module_type][token_idx])
                    elif metric_name == "mse" and mse and token_idx in mse[layer][module_type]:
                        layers.append(layer)
                        values.append(mse[layer][module_type][token_idx])
                if not values:
                    continue
                plt.subplot(1, 2, i + 1)
                plt.plot(layers, values, marker="o")
                plt.xlabel("Layer")
                plt.ylabel(metric_name.upper())
                plt.title(f"{module_type.upper()} {metric_name.upper()} - Token {token_idx}")
                plt.grid(True)
            plt.suptitle(f"{model_name} {metric_name.upper()} - Token {token_idx}", fontsize=14)
            wandb.log({f"{model_name}/{metric_name}_token{token_idx}": wandb.Image(plt)}, step=step)
            plt.close()
            time.sleep(0.1)

        # === Heatmap 可视化，每层一个图（防卡死）
        for module_type in ["attention", "ffn"]:
            for layer in sorted(jacobian.keys()):
                if token_idx in jacobian[layer][module_type]:
                    mat = jacobian[layer][module_type][token_idx]
                    if mat.shape[0] > 100:
                        continue  # 避免太大卡死
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(mat, cmap="viridis", cbar=True)
                    plt.title(f"{module_type.upper()} Heatmap - Layer {layer} - Token {token_idx}")
                    wandb.log({
                        f"{model_name}/{module_type}_heatmap_layer{layer}_token{token_idx}": wandb.Image(plt)
                    }, step=step)
                    plt.close()
