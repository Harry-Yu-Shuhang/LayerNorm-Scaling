import os
import numpy as np
import torch
from tqdm import tqdm
from peft_pretraining.modeling_llama import LlamaRMSNorm

class JacobianCalculator:
    def __init__(self, output_dir="results/Jacobian"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.error_log_path = os.path.join(self.output_dir, "error.log")
        with open(self.error_log_path, "w") as f:
            f.write("")

    def _log_error(self, msg):
        with open(self.error_log_path, "a") as f:
            f.write(msg + "\n")
        print(f"🛑 {msg}")

    def compute_jacobian(self, model, model_name, step, input_ids, attention_mask):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            print(f"🔕 Rank {torch.distributed.get_rank()} 不保存 Jacobian，跳过。")
            return {}, {}

        device = next(model.parameters()).device
        print(f"\n🟢 Step {step} - 在 {device} 上计算 Jacobian")
        print(f"🔍 注册 LayerNorm (RMSNorm) Hook")

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        input_embeddings = model.get_input_embeddings()(input_ids)
        input_embeddings.requires_grad_()

        norm_inputs = {}
        def make_hook_fn(layer_index, tag):
            def hook_fn(module, input, output):
                key = f"layer_{layer_index}_{tag}"
                try:
                    norm_inputs[key] = input[0].detach().clone().requires_grad_()
                except Exception as e:
                    self._log_error(f"Hook 捕获失败: {key} - 错误信息: {e}")
            return hook_fn

        handles = []
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'input_layernorm'):
                handles.append(layer.input_layernorm.register_forward_hook(make_hook_fn(i, "input")))
            if hasattr(layer, 'post_attention_layernorm'):
                handles.append(layer.post_attention_layernorm.register_forward_hook(make_hook_fn(i, "post")))

        outputs = model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        for h in handles:
            h.remove()

        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states) - 1
        hidden_dim = hidden_states[1].shape[-1]
        seq_len = attention_mask[0].sum().item()
        selected_tokens = list(range(seq_len))

        print(f"📏 token 总数: {input_ids.shape[1]}, 非 pad token: {seq_len}, 隐层维度: {hidden_dim}, 层数: {num_layers}")
        jacobian_results = {}
        frobenius_results = {}
        mse_results = {}

        for layer in tqdm(range(0, num_layers), desc=f"Step {step} - Jacobian", unit="layer"):
            ln_key = f"layer_{layer}_input"
            if ln_key not in norm_inputs:
                self._log_error(f"⛔️ 未捕获 Layer {layer} 的 RMSNorm 输入 ({ln_key})，跳过")
                continue

            ln_output = norm_inputs[ln_key]
            if not ln_output.requires_grad:
                self._log_error(f"⚠️ Layer {layer} 的 RMSNorm 输入不支持梯度计算 (requires_grad=False)，跳过")
                continue

            layer_jacobians = {"attention": {}, "ffn": {}}
            frob_layer = {"attention": {}, "ffn": {}}
            mse_layer = {"attention": {}, "ffn": {}}

            for token_idx in selected_tokens:
                try:
                    attn_output = hidden_states[layer + 1][:, token_idx, :]
                    print(f"🔎 Layer {layer}, Token {token_idx}, Attention 输出 shape: {attn_output.shape}")
                    if attn_output.grad_fn is None:
                        self._log_error(f"❗️ Layer {layer}, Token {token_idx} Attention 无 grad_fn，跳过")
                        continue

                    jacobian_attn = self._compute_single_jacobian(attn_output, ln_output, token_idx, layer, "attention")
                    if jacobian_attn is not None:
                        layer_jacobians["attention"][token_idx] = jacobian_attn
                        frob_layer["attention"][token_idx] = np.linalg.norm(jacobian_attn, ord="fro")
                        mse_layer["attention"][token_idx] = np.mean(jacobian_attn ** 2)

                    if layer + 2 < len(hidden_states):
                        ffn_output = hidden_states[layer + 2][:, token_idx, :]
                        print(f"🔎 Layer {layer}, Token {token_idx}, FFN 输出 shape: {ffn_output.shape}")
                        if ffn_output.grad_fn is None:
                            self._log_error(f"❗️ Layer {layer}, Token {token_idx} FFN 无 grad_fn，跳过")
                            continue

                        jacobian_ffn = self._compute_single_jacobian(ffn_output, ln_output, token_idx, layer, "ffn")
                        if jacobian_ffn is not None:
                            layer_jacobians["ffn"][token_idx] = jacobian_ffn
                            frob_layer["ffn"][token_idx] = np.linalg.norm(jacobian_ffn, ord="fro")
                            mse_layer["ffn"][token_idx] = np.mean(jacobian_ffn ** 2)

                except Exception as e:
                    self._log_error(f"❌ 计算出错 - Layer {layer}, Token {token_idx}：{e}")

            if layer_jacobians["attention"] or layer_jacobians["ffn"]:
                jacobian_results[layer] = layer_jacobians
                frobenius_results[layer] = frob_layer
                mse_results[layer] = mse_layer

        with open(self.error_log_path, "r") as f:
            if jacobian_results and len(f.readlines()) == 0:
                save_path = os.path.join(self.output_dir, f"{model_name}_step_{step}_jacobian.npz")
                np.savez_compressed(save_path,
                                    jacobian=jacobian_results,
                                    frobenius=frobenius_results,
                                    mse=mse_results)
                print(f"✅ Jacobian 已保存: {save_path}")
                return frobenius_results, mse_results
            else:
                print("🚫 检测到错误或结果为空，未保存")
                return {}, {}

    def _compute_single_jacobian(self, token_output, ln_output, token_idx, layer_idx, tag):
        jacobian = []
        hidden_dim = token_output.shape[-1]

        # 取第一个 batch（假设 batch_size=1，否则你得改成对所有 batch 求和或平均）
        token_output = token_output[0]  # [hidden_dim]
        for dim in range(hidden_dim):
            grad_outputs = torch.zeros_like(token_output)
            grad_outputs[dim] = 1.0
            try:
                grads = torch.autograd.grad(
                    outputs=token_output,
                    inputs=ln_output,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            except Exception as e:
                self._log_error(f"🧨 grad 出错 - Layer {layer_idx}, Token {token_idx}, Dim {dim} ({tag}) - {e}")
                return None

            if grads is None:
                self._log_error(f"🚫 grad 为 None - Layer {layer_idx}, Token {token_idx}, Dim {dim} ({tag})")
                continue

            try:
                grad_tensor = grads[0, token_idx, :]  # only 1st batch
            except IndexError as e:
                self._log_error(f"📛 grad 取 token_idx 错误 - Layer {layer_idx}, Token {token_idx}, Dim {dim} ({tag}) - {e}")
                return None

            grad_tensor = torch.nan_to_num(grad_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            jacobian.append(grad_tensor.detach().cpu().numpy())

        if len(jacobian) == 0:
            self._log_error(f"📭 所有 grad 为 None - Layer {layer_idx}, Token {token_idx} ({tag})")
            return None

        return np.stack(jacobian, axis=0)  # shape: [hidden_dim, hidden_dim]

