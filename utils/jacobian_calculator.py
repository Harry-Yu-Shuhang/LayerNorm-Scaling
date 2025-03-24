import torch
import numpy as np
import os
from tqdm import tqdm
from exp_config.utils import Utils

class JacobianCalculator:
    def __init__(self):
        self.utils = Utils()
        self.results_dir = os.path.join(self.utils.output_dir, "Jacobian_Debug")
        os.makedirs(self.results_dir, exist_ok=True)

    def compute_jacobian(self, model, model_name, step, input_ids, attention_mask):
        device = next(model.parameters()).device
        print(f"🟢 Step {step} - 在 {device} 上计算 Jacobian")

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        input_embeddings = model.get_input_embeddings()(input_ids)
        input_embeddings.requires_grad_()

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

        for layer in tqdm(range(1, num_layers), desc=f"Step {step} - Jacobian", unit="layer"):
            print(f"\n🔹 Layer {layer}:")
            layer_jacobians = {"attention": {}, "ffn": {}}

            ln_output = hidden_states[layer - 1].clone().detach().requires_grad_()
            if not ln_output.requires_grad:
                print(f"❌ Layer {layer}: ln_output.requires_grad=False，跳过")
                continue

            for token_idx in selected_tokens:
                try:
                    attn_output = hidden_states[layer][:, token_idx, :].squeeze(0)
                    jacobian_attn = self.compute_gradients(attn_output, ln_output, token_idx, hidden_dim, layer, model, mode="attention")
                    if jacobian_attn is not None:
                        layer_jacobians["attention"][token_idx] = jacobian_attn

                    if layer + 1 < num_layers:
                        ffn_output = hidden_states[layer + 1][:, token_idx, :].squeeze(0)
                        jacobian_ffn = self.compute_gradients(ffn_output, ln_output, token_idx, hidden_dim, layer, model, mode="ffn")
                        if jacobian_ffn is not None:
                            layer_jacobians["ffn"][token_idx] = jacobian_ffn

                except Exception as e:
                    print(f"❌ Layer {layer}, Token {token_idx} 错误: {e}")

            if layer_jacobians["attention"] or layer_jacobians["ffn"]:
                jacobian_results[layer] = layer_jacobians

        if jacobian_results:
            save_path = os.path.join(self.utils.output_dir, f"{model_name}_step_{step}_jacobian.npz")
            np.savez_compressed(save_path, jacobian=jacobian_results)
            print(f"✅ Jacobian 保存至 {save_path}")
        else:
            print("⚠️ Jacobian 为空，跳过保存")

    def compute_gradients(self, token_output, ln_output, token_idx, hidden_dim, layer, model, mode="attention"):
        jacobian = []
        failed_dims = []

        layer_dir = os.path.join(self.results_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)
        path_file = os.path.join(layer_dir, f"token_{token_idx}.txt")

        with open(path_file, "w") as f:
            f.write(f"🔍 Token {token_idx} 的 {mode} 梯度计算调试信息\n")
            f.write(f"📎 token_output.shape: {token_output.shape}\n")
            f.write(f"📎 token_output.grad_fn: {token_output.grad_fn}\n")

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
                failed_dims.append(dim)
                with open(path_file, "a") as f:
                    f.write(f"❌ dim {dim} 的 grad 为 None，token_output.grad_fn = {token_output.grad_fn}\n")
                continue

            grads = grads[:, token_idx, :]
            grads = torch.nan_to_num(grads, nan=0.0, posinf=1.0, neginf=-1.0)
            jacobian.append(grads.detach().cpu().numpy().squeeze())

        with open(path_file, "a") as f:
            if failed_dims:
                f.write(f"⚠️ 计算失败的维度: {failed_dims}\n")
            else:
                f.write(f"✅ 所有维度计算成功\n")

        return np.stack(jacobian, axis=0) if jacobian else None

from transformers import TrainerCallback

class JacobianCallback(TrainerCallback):
    """在每个 epoch 结束时计算 Jacobian"""
    def __init__(self, jacobian_calculator, model_name, tokenizer):
        super().__init__()
        self.jacobian_calculator = jacobian_calculator
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            print("❌ Trainer 未初始化，跳过 Jacobian 计算")
            return

        print(f"🟢 Epoch {state.epoch:.0f} 结束，开始计算 Jacobian...")

        dataset = self.trainer.train_dataset
        sample = next(iter(dataset))

        input_ids = sample["input_ids"].unsqueeze(0)  # 加 batch 维度
        attention_mask = sample["attention_mask"].unsqueeze(0)

        decoded_text = self.tokenizer.decode(
            input_ids[0, attention_mask[0].bool()],
            skip_special_tokens=True
        )
        print(f"📌 Jacobian 计算文本: {decoded_text}")
        print(f"📌 input_ids.shape: {input_ids.shape}, mask.shape: {attention_mask.shape}")

        self.jacobian_calculator.compute_jacobian(
            model=self.trainer.model,
            model_name=self.model_name,
            step=int(state.epoch),
            input_ids=input_ids,
            attention_mask=attention_mask
        )
