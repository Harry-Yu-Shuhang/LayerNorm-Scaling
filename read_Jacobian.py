import numpy as np

file = "results/Jacobian/9m_res_pre_lr5e-4_step_1000_jacobian.npz"
data = np.load(file, allow_pickle=True)

jacobian = data["jacobian"].item()
frobenius = data["frobenius"].item()
mse = data["mse"].item()

print(f"✅ 成功加载: {file}")
print(f"📏 总共 {len(jacobian)} 层")

# 示例输出某一层某个 token 的值
layer_id = 0
token_id = 0
if layer_id in frobenius:
    print(f"\n🔍 Layer {layer_id}, Token {token_id} Attention Frobenius:", frobenius[layer_id]["attention"][token_id])
    print(f"🔍 Layer {layer_id}, Token {token_id} FFN Frobenius:", frobenius[layer_id]["ffn"].get(token_id, "❌ 无该 token"))

if layer_id in mse:
    print(f"📉 MSE:", mse[layer_id]["attention"][token_id])
