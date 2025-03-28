# utils/train_full_model.py

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft_pretraining.modeling_llama import LlamaForCausalLM
from peft_pretraining import training_utils

from utils.train_module import train_model

from peft_pretraining.dataloader import PreprocessedIterableDataset
from transformers import AutoTokenizer
import datasets
import torch.distributed as dist

def load_train_dataset(tokenizer, args, rank, world_size):
    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    data = data.shuffle(seed=32)

    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(data, rank=rank, world_size=world_size)

    return PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)


def train_full_model(args):
    assert "LOCAL_RANK" in os.environ
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    device = f"cuda:{local_rank}"

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    dataset = load_train_dataset(tokenizer, args, global_rank, world_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4)

    config = AutoConfig.from_pretrained(args.model_config)
    model = LlamaForCausalLM(config).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    run_config = vars(args)
    global_step = update_step = tokens_seen = tokens_seen_before = 0
    pad_idx = tokenizer.pad_token_id
    pbar = tqdm(total=args.total_batch_size, desc="Training", ncols=100) if global_rank == 0 else None

    update_step, global_step, tokens_seen, tokens_seen_before = train_model(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device,
        args=args,
        scheduler=scheduler,
        optimizer=optimizer,
        run_config=run_config,
        global_rank=global_rank,
        local_rank=local_rank,
        pad_idx=pad_idx,
        update_step=update_step,
        global_step=global_step,
        tokens_seen=tokens_seen,
        tokens_seen_before=tokens_seen_before,
        layer_wise_flag=False,
        evaluate_model=training_utils.evaluate_model,
        preprocess_batched=training_utils.get_preprocess_fn(tokenizer, args.max_length),
        pbar=pbar
    )
