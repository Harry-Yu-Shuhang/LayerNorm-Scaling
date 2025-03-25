#微调huggingface需要
# from transformers import TrainerCallback

# class JacobianCallback(TrainerCallback):
#     """在每个 epoch 结束时计算 Jacobian"""
#     def __init__(self, jacobian_calculator, model_name, tokenizer, max_tokens=256):
#         super().__init__()
#         self.jacobian_calculator = jacobian_calculator
#         self.model_name = model_name
#         self.tokenizer = tokenizer
#         self.trainer = None
#         self.max_tokens = max_tokens

#     def set_trainer(self, trainer):
#         self.trainer = trainer

#     def on_epoch_end(self, args, state, control, **kwargs):
#         if self.trainer is None:
#             print("❌ Trainer 未初始化，跳过 Jacobian 计算")
#             return

#         print(f"🟢 Epoch {state.epoch:.0f} 结束，开始计算 Jacobian...")

#         # 获取样本（只取第一个，避免批量过大）
#         dataset = self.trainer.train_dataset
#         sample = next(iter(dataset))

#         input_ids = sample["input_ids"].unsqueeze(0)
#         attention_mask = sample["attention_mask"].unsqueeze(0)

#         # 防止太长导致计算爆炸
#         if input_ids.shape[1] > self.max_tokens:
#             input_ids = input_ids[:, :self.max_tokens]
#             attention_mask = attention_mask[:, :self.max_tokens]

#         decoded_text = self.tokenizer.decode(
#             input_ids[0, attention_mask[0].bool()],
#             skip_special_tokens=True
#         )
#         print(f"📌 Jacobian 计算文本: {decoded_text}")
#         print(f"📌 input_ids.shape: {input_ids.shape}, mask.shape: {attention_mask.shape}")

#         self.jacobian_calculator.compute_jacobian(
#             model=self.trainer.model,
#             model_name=self.model_name,
#             step=int(state.epoch),
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
