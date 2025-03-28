# utils/train_module.py

import os
import torch
import json
from tqdm import tqdm
from loguru import logger
from utils.jacobian_calculator import JacobianCalculator
import time


def train_model(
    model, tokenizer, dataloader, device, args, scheduler, optimizer,
    run_config, global_rank, local_rank,
    pad_idx, update_step, global_step,
    tokens_seen, tokens_seen_before,
    layer_wise_flag, evaluate_model, preprocess_batched, pbar
):
    world_size = torch.distributed.get_world_size()
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"
    update_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        global_step += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item()

        loss = model(**batch, labels=labels).loss
        (loss / args.gradient_accumulation).backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        update_step += 1

        if global_rank == 0 and pbar: pbar.update(1)

        if update_step % args.eval_every == 0 and global_rank == 0:
            logger.info(f"Running evaluation at step {update_step}")
            eval_loss, _ = evaluate_model(args, model, preprocess_batched, pad_idx, global_rank,
                                          torch.distributed.get_world_size(), device, args.batch_size)

            logger.info(f"Eval loss = {eval_loss}")
            try:
                sample = next(iter(dataloader))
                input_ids = sample["input_ids"].to(device)
                attention_mask = sample["attention_mask"].to(device)

                jacobian_calculator = JacobianCalculator()
                jacobian_calculator.compute_jacobian(
                    model=model.module if not args.single_gpu else model,
                    model_name=args.run_name,
                    step=update_step,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            except Exception as e:
                logger.error(f"Jacobian 计算失败: {e}")

    return update_step, global_step, tokens_seen, tokens_seen_before
