import os
import numpy as np
import torch
from tqdm import tqdm

class JacobianCalculator:
    def __init__(self, output_dir="results/Jacobian"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ln_inputs = {}  # 🆕 用于保存 Hook 捕获的输入

    def compute_jacobian(self, model, model_name, step, input_ids, attention_mask):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            print(f"🔕 Rank {torch.distributed.get_rank()} 不保存 Jacobian，跳过。")
            return {}, {}

        device = next(model.parameters()).device
        print(f"🟢 Step {step} - 在 {device} 上计算 Jacobian")

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        input_embeddings = model.get_input_embeddings()(input_ids)
        input_embeddings.requires_grad_()

        # 🧩 注册 forward hook，获取 LayerNorm 的输入
        self._register_ln_hooks(model)

        outputs = model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states
        num_tokens = input_ids.shape[1]
        hidden_dim = hidden_states[1].shape[-1]
        num_layers = len(hidden_states) - 1
        seq_len = attention_mask[0].sum().item()
        selected_tokens = list(range(seq_len))

        print(f"📏 输入 token 数: {num_tokens}, 非 pad token: {seq_len}, 隐层维度: {hidden_dim}, 总层数: {num_layers}")
        jacobian_results = {}
        frobenius_results = {}
        mse_results = {}

        for layer in tqdm(range(1, num_layers), desc=f"Step {step} - Jacobian", unit="layer"):
            layer_jacobians = {"attention": {}, "ffn": {}}
            frob_layer = {"attention": {}, "ffn": {}}
            mse_layer = {"attention": {}, "ffn": {}}

            ln_output = self.ln_inputs.get(layer - 1, None)
            if ln_output is None:
                print(f"⛔️ Layer {layer}: 未捕获到 LayerNorm 输入，跳过")
                continue
            else:
                print(f"✅ Layer {layer}: 捕获到 LayerNorm 输入，requires_grad={ln_output.requires_grad}")

            for token_idx in selected_tokens:
                try:
                    attn_output = hidden_states[layer][:, token_idx, :].squeeze(0)
                    jacobian_attn = self._compute_single_jacobian(attn_output, ln_output, token_idx)
                    if jacobian_attn is not None:
                        layer_jacobians["attention"][token_idx] = jacobian_attn
                        frob_layer["attention"][token_idx] = np.linalg.norm(jacobian_attn, ord="fro")
                        mse_layer["attention"][token_idx] = np.mean(jacobian_attn ** 2)

                    if layer + 1 < num_layers:
                        ffn_output = hidden_states[layer + 1][:, token_idx, :].squeeze(0)
                        jacobian_ffn = self._compute_single_jacobian(ffn_output, ln_output, token_idx)
                        if jacobian_ffn is not None:
                            layer_jacobians["ffn"][token_idx] = jacobian_ffn
                            frob_layer["ffn"][token_idx] = np.linalg.norm(jacobian_ffn, ord="fro")
                            mse_layer["ffn"][token_idx] = np.mean(jacobian_ffn ** 2)

                except Exception as e:
                    print(f"❌ Layer {layer}, Token {token_idx} 错误: {e}")

            if layer_jacobians["attention"] or layer_jacobians["ffn"]:
                jacobian_results[layer] = layer_jacobians
                frobenius_results[layer] = frob_layer
                mse_results[layer] = mse_layer

        if jacobian_results:
            save_path = os.path.join(self.output_dir, f"{model_name}_step_{step}_jacobian.npz")
            np.savez_compressed(save_path,
                                jacobian=jacobian_results,
                                frobenius=frobenius_results,
                                mse=mse_results)
            print(f"✅ Jacobian 保存至 {save_path}")
            return frobenius_results, mse_results
        else:
            print("⚠️ Jacobian 为空，跳过保存")
            return {}, {}

    def _compute_single_jacobian(self, token_output, ln_output, token_idx):
        jacobian = []
        hidden_dim = token_output.shape[-1]
        for dim in range(hidden_dim):
            grad_outputs = torch.zeros_like(token_output)
            grad_outputs[dim] = 1.0

            grads = torch.autograd.grad(
                outputs=token_output,
                inputs=ln_output,
                grad_outputs=grad_outputs,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if grads is None:
                print(f"🚫 Grad 为 None - Token {token_idx} at dim {dim} of layer")
                return None

            grads = grads[:, token_idx, :]
            grads = torch.nan_to_num(grads, nan=0.0, posinf=1.0, neginf=-1.0)
            jacobian.append(grads.detach().cpu().numpy().squeeze())

        return np.stack(jacobian, axis=0)

    def _register_ln_hooks(self, model):
        """
        为 LlamaDecoderLayer 中的 input_layernorm 注册 hook，记录每层 LayerNorm 的输入。
        """

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                self.ln_inputs[layer_idx] = input[0].detach().requires_grad_()
            return hook_fn

        self.ln_inputs = {}  # 清空旧缓存

        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'input_layernorm'):
                layer.input_layernorm.register_forward_hook(make_hook(i))
            else:
                print(f"⚠️ 第 {i} 层没有 input_layernorm，跳过 hook")
