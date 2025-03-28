import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def visualize_and_log_to_wandb(model_name, step, tokens, project_dir="results/Jacobian"):
    if int(os.environ.get("RANK", "0")) != 0:
        return  # 🚫 非主进程，不执行 wandb.log()

    """
    从 .npz 文件中加载 Jacobian，并将指定 tokens 的 Frobenius 和 heatmap 图像上传到 wandb。
    """
    file_path = os.path.join(project_dir, f"{model_name}_step_{step}_jacobian.npz")
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    jacobian = data["jacobian"].item()
    frobenius = data["frobenius"].item()

    for token_idx in tokens:
        for module_type in ["attention", "ffn"]:
            # Frobenius 折线图
            layers = []
            frob_vals = []
            for layer in sorted(frobenius.keys()):
                if token_idx in frobenius[layer][module_type]:
                    layers.append(layer)
                    frob_vals.append(frobenius[layer][module_type][token_idx])

            if frob_vals:
                plt.figure(figsize=(8, 4))
                plt.plot(layers, frob_vals, marker="o", label=f"{module_type.upper()} Token {token_idx}")
                plt.xlabel("Layer")
                plt.ylabel("Frobenius Norm")
                plt.title(f"{module_type.upper()} Frobenius Norm - Token {token_idx}")
                plt.grid(True)
                plt.legend()
                wandb.log({f"{module_type}_frobenius_token{token_idx}": wandb.Image(plt)}, step=step)
                plt.close()

            # Heatmap 可视化（每层一个）
            for layer in sorted(jacobian.keys()):
                if token_idx in jacobian[layer][module_type]:
                    mat = jacobian[layer][module_type][token_idx]
                    if mat.shape[0] > 100:  # 防止太大卡死
                        continue
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(mat, cmap="viridis", cbar=True)
                    plt.title(f"{module_type.upper()} Heatmap - Layer {layer} - Token {token_idx}")
                    wandb.log({f"{module_type}_heatmap_layer{layer}_token{token_idx}": wandb.Image(plt)}, step=step)
                    plt.close()
