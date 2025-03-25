import yaml
import os
import torch
import random
import numpy as np
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# ✅ 显式禁用 reentrant checkpointing
import torch.utils.checkpoint as checkpoint

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class Utils:
    def __init__(self, config_path="exp_config/conf.yaml"):
        # 读取 YAML 配置文件
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 解析不同类别的配置
        self.experiment_config = config["experiment"]
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.dataset_config = config["dataset"]

        # **实验参数**
        self.seed = self.experiment_config["seed"]
        self.selected_tokens = self.experiment_config["selected_tokens"]

        # **模型参数**
        self.model_names = self.model_config["model_names"]

        # **训练参数**
        self.num_epochs = self.training_config["num_epochs"]
        self.per_device_batch_size = self.training_config["per_device_batch_size"]
        self.output_dir = self.training_config["output_dir"]
        self.logging_steps = self.training_config["logging_steps"]
        self.learning_rate = float(self.training_config["learning_rate"]) 
        self.weight_decay = float(self.training_config["weight_decay"])    
        self.max_grad_norm = float(self.training_config["max_grad_norm"])  
        self.max_steps = int(self.training_config["max_steps"])
        self.warmup_ratio = float(self.training_config["warmup_ratio"])  # ✅ 添加 warmup
        self.gradient_accumulation_steps = int(self.training_config["gradient_accumulation_steps"])  # ✅ 添加梯度累积


        # **数据集参数**
        self.dataset_variant = self.dataset_config["dataset_variant"]

        # 获取环境变量中的 DDP 参数
        self.rank = int(os.environ.get("RANK", 0))  # 进程编号
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 本地 GPU 进程编号
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))  # 进程总数

        # **初始化分布式训练**
        if self.world_size > 1:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank
            )
            torch.cuda.set_device(self.local_rank)  # 绑定当前进程到 GPU
            print(f"✅ 进程 {self.rank}/{self.world_size} 绑定到 GPU {self.local_rank}")

        # **设置设备**
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        # 设置随机种子
        self.set_seed(self.seed)

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 放在 __init__ 中
        if hasattr(checkpoint, "_set_checkpoint_engine"):
            try:
                checkpoint._set_checkpoint_engine("non_reentrant")
                print("✅ checkpoint 引擎设置为 non_reentrant")
            except Exception as e:
                print(f"⚠️ 设置 checkpoint 引擎失败: {e}")


    def set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_c4_data(self):
        """加载 C4 数据集"""
        print(f"🔵 加载 C4 数据集 ({self.dataset_variant})")
        download_config = DownloadConfig(
            resume_download=True,
            max_retries=6,  # 如果超时，最多重试 6 次
            storage_options={"timeout": 600}  # 设置 10 分钟超时，避免太快断开
        )

        dataset = load_dataset(
            "allenai/c4", 
            self.dataset_variant, 
            split="train", 
            streaming=True,
            data_files="en/c4-train.0000*-of-01024.json.gz",
            download_config=download_config,
        )

        return dataset


    def load_tokenizer_and_model(self, model_name):
        """加载 tokenizer 和 Transformer 模型（适配 checkpoint & DDP）"""
        print(f"🔵 正在加载模型 {model_name} 到 {self.device}...")

        # ✅ 设置全局 checkpoint 引擎（必须在模型加载前设置）
        if hasattr(checkpoint, "_set_checkpoint_engine"):
            try:
                checkpoint._set_checkpoint_engine("non_reentrant")
                print("✅ 设置 torch.utils.checkpoint 引擎为 non_reentrant")
            except Exception as e:
                print(f"⚠️ 设置 checkpoint 引擎失败: {e}")
        else:
            print("⚠️ 当前 PyTorch 版本不支持 _set_checkpoint_engine")

        # ✅ 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # DeepSeek 兼容性 fix

        # ✅ 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None  # 让 .to(self.device) 生效
        ).to(self.device)

        # ✅ 设置配置项
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(use_reentrant=False)
            print("✅ 启用 gradient checkpointing，且禁用 reentrant")
        except TypeError:
            model.gradient_checkpointing_enable()
            print("⚠️ 当前 transformers 不支持 use_reentrant 参数，启用默认 checkpointing")

        print(f"✅ use_cache: {model.config.use_cache}")
        print(f"✅ gradient checkpointing: {model.is_gradient_checkpointing}")

        # ✅ 适配 DDP 多卡训练
        if torch.cuda.device_count() > 1 and torch.distributed.is_initialized():
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            print(f"✅ 模型使用 DDP 分布式包裹 (GPU {self.local_rank})")

        print(f"✅ {model_name} 加载完成！")
        return tokenizer, model

    def get_training_args(self):
        """返回 Hugging Face Trainer 的训练参数"""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.per_device_batch_size,
            learning_rate=self.learning_rate,  
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            warmup_ratio=self.warmup_ratio,  # ✅ 添加 warmup
            gradient_accumulation_steps=self.gradient_accumulation_steps,  # ✅ 添加梯度累积
            max_steps=self.max_steps,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=self.logging_steps,
            save_total_limit=2,
            report_to="none",
            gradient_checkpointing=False,  # ✅ 显式禁用，防止覆盖 model.gradient_checkpointing_enable()
            bf16=torch.cuda.is_bf16_supported(),  # 兼容性更强
            fp16=not torch.cuda.is_bf16_supported(),
            optim="adamw_torch",  # ✅ 使用 `adamw_torch` 兼容 `accelerate`
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,  # ✅ 避免 DDP 设备问题
            dataloader_pin_memory=False,  # ✅ 避免内存泄漏
            torch_compile=False,  # ✅ 避免 `torch.compile` 引发的设备冲突
            ddp_backend="nccl" if self.world_size > 1 else None,  # ✅ 适配 DDP
        )