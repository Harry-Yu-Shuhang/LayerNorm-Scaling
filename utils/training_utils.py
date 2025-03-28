# utils/training_utils.py

from datasets import load_dataset, distributed
from peft_pretraining.dataloader import PreprocessedIterableDataset

def load_train_dataset(tokenizer, args, global_rank, world_size):
    data = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    data = data.shuffle(seed=42)
    if not args.single_gpu:
        data = distributed.split_dataset_by_node(data, rank=global_rank, world_size=world_size)
    return PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)

def load_eval_data(name, world_size, rank):
    val = load_dataset(name, "en", split="validation", streaming=True, trust_remote_code=True)
    val = val.shuffle(seed=42)
    return distributed.split_dataset_by_node(val, rank=rank, world_size=world_size)

def get_preprocess_fn(tokenizer, max_length):
    def preprocess_batched(batch):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
    return preprocess_batched

def batch_fn(streaming_dataset, batch_size):
    batch = []
    for example in streaming_dataset:
        batch.append(example)
        if len(batch) >= batch_size:
            yield batch
            batch = []
