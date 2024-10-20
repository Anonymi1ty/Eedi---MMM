import sys
import logging

import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

"""
A simple example on using SFTTrainer and Accelerate to finetune Phi-3 models. For
more advanced example, please follow HF alignment-handbook/scripts/run_sft.py.

"""

# 获取日志记录器
logger = logging.getLogger(__name__)

###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,  # 使用bfloat16精度进行训练
    "do_eval": False,  # 是否进行评估
    "learning_rate": 5.0e-06,  # 学习率
    "log_level": "info",  # 日志记录级别
    "logging_steps": 20,  # 每20步记录一次日志
    "logging_strategy": "steps",  # 日志记录策略
    "lr_scheduler_type": "cosine",  # 使用cosine学习率调度器
    "num_train_epochs": 1,  # 训练的epoch数量
    "max_steps": -1,  # 最大训练步数（-1表示不限步数）
    "output_dir": "./checkpoint_dir",  # 模型保存目录
    "overwrite_output_dir": True,  # 是否覆盖输出目录
    "per_device_eval_batch_size": 4,  # 每个设备上的评估batch大小
    "per_device_train_batch_size": 4,  # 每个设备上的训练batch大小
    "remove_unused_columns": True,  # 是否移除未使用的列
    "save_steps": 100,  # 每100步保存一次模型
    "save_total_limit": 1,  # 保存的模型数量限制
    "seed": 0,  # 随机种子
    "gradient_checkpointing": True,  # 使用梯度检查点减少内存消耗
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,  # 梯度累积步数
    "warmup_ratio": 0.2,  # 学习率预热比例
    }

# 配置LoRA (Low-Rank Adaptation) 参数
peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}

# 初始化训练参数配置和LoRA配置
train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# 设置日志级别
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# 记录一些训练过程的信息
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")

################
# Model Loading
################

# 加载Phi-3.5-mini-instruct模型
checkpoint_path = "microsoft/Phi-3.5-mini-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,  # 信任远程代码来加载模型
    attn_implementation="flash_attention_2",  # 使用flash attention进行注意力计算
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)  # 从huggingface加载模型和分词器
# **需要从HuggingFace加载的部分**

# 设置分词器的最大序列长度、填充token和填充方向
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token
# 使用unk_token防止生成无限长的内容
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

##################
# Data Processing
##################
# 定义一个数据处理函数，用于应用对话模板
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

# 加载数据集并进行处理
raw_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")  # **需要从HuggingFace加载的部分**
train_dataset = raw_dataset["train_sft"]
test_dataset = raw_dataset["test_sft"]
column_names = list(train_dataset.features)

# 对训练集进行模板应用处理
processed_train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to train_sft",
)

# 对测试集进行模板应用处理
processed_test_dataset = test_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to test_sft",
)

###########
# Training
###########
# 初始化SFTTrainer（指令微调）并进行训练
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    max_seq_length=2048,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True
)
# 开始训练
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

#############
# Evaluation
#############
tokenizer.padding_side = 'left'  # 设置填充方向为左
metrics = trainer.evaluate()  # 评估模型
metrics["eval_samples"] = len(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

############
# Save model
############
trainer.save_model(train_conf.output_dir)  # 保存模型
