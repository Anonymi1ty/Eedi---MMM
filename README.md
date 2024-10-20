---
license: mit
license_link: https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/LICENSE
language:
- multilingual
pipeline_tag: text-generation
tags:
- nlp
- code
widget:
- messages:
  - role: user
    content: Can you provide ways to eat combinations of bananas and dragonfruits?
library_name: transformers
---

---

>缺少`model-00002-of-00002.safetensors`,`model-00001-of-00002.safetensors`三个文件
>
>自己在 [model-00002-of-00002.safetensors...下载](https://www.kaggle.com/models/richolson/phi-3.5-mini-instruct/PyTorch/default?select=model-00001-of-00002.safetensors)

使用说明详见[Phi-3](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)

---

> 环境需要：

参考[ Phi-3.5-mini-instruct：](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)

Phi-3 family has been integrated in the `4.43.0` version of `transformers`. The current `transformers` version can be verified with: `pip list | grep transformers`.

Examples of required packages:

<!--需要什么微调方法下载什么库-->

`trl(SFT)`,`peft(LoRA)`

```
flash_attn==2.5.8
torch==2.3.1
accelerate==0.31.0
transformers==4.43.0
peft
trl
```

> 示例解析

文件`sample_finetune.py`主要使用 ``SFTTrainer` 和` Accelerate`进行指令微调，并且对使用`LoRA（Low-Rank Adaptation）`冻结大部分权重矩阵

示例使用**ultrachat_200k（对话数据集，主要用于增强模型在多轮对话中的理解和生成能力。）**作为加载数据，这个大型数据集可能并不会使本次比赛的性能变得更好，尽管Phi-3.5被设计成重点用于推理和数学逻辑。

这个是可能替代的数据集[MalAlgoQA](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/541222)/手动数据构建

> kaggle运行

**指令：**

对于 Python，您可以通过在命令行中预置 `！`添加到代码单元格中。For instance, to install a new package using pip, run `!pip install my-new-package`.

**离线环境：**

将笔记本配置为在禁用 Internet 的提交期间通过使用 Package Manager 编辑器执行脱机 pip 安装。然后，这些笔记本可以提交到互联网禁用竞赛中。

在包管理器编辑器中，输入 pip install 命令，例如 `pip install my-new-package`。您还可以通过添加 `pip install -U my-existing-package` 来升级现有软件包。您还可以使用 pip 通过 `pip install git+https://github.com/author/package.git` .





> 可能遇到的问题：

Kaggle上安装flash-attn在安装时时间很久(大概需要3.5h)（可能可以找下面方法解决？)

![image-20241020133209964](D:\Typora\myimage\image-20241020133209964.png)