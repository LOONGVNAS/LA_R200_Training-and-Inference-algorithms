## 版权声明

本文档转自公众号： AI云智工坊。若有侵权，请及时联系本人进行删除。

## 简介

大模型LLM量化技术是模型资源（显存和计算复杂度）优化最简单和直接方案。

vLLM提供的模型量化工具llm-compressor可完成对Qwen3-32B BF16精度的模型进行INT8 W8A8（权重8位，激活8位）量化。

<img width="1080" height="608" alt="image" src="https://github.com/user-attachments/assets/8095c40d-fdbd-4c93-b1e5-84047cd8a2a9" />

想要在资源受限的设备上部署大模型推理服务，同时还要保留其精度和性能，通常采用模型量化技术来减少模型权重和激活对显存的占用和计算需求。


## 📖 文章目录

- • 1 什么是模型量化

- • 2 量化的类型（PTQ和QAT）

- • 2.1 训练后量化（PTQ）

- • 2.2 量化感知训练（QAT）

- • 3 常用的量化技术

- • 4 量化位宽选择

- • 5 LLM Compressor量化工具集

- • 6 使用LLM Compressor量化Qwen3-32B

- • 6.1 安装llm-compressor

- • 6.2 下载模型权重文件

- • 6.3 准备校验数据

- • 6.4 执行量化

- • 6.5 评估准确性

- • 7、量化遇到问题

- • 7.1 校准数据集下载问题

- • 7.2 GSM-8K数据集下载问题

- • 7.3 显存不足导致模型load失败

- • 8 部署量化后Qwen3-32B

- • 总结

## **1、什么是模型量化**

简单来说模型量化是一种模型压缩技术，在保持模型性能和精度的前提下，通过降低模型参数的数值精度（从FP32转换为FP8或者INT8或者INT4）来减少模型的显存占用和计算开销。

比如对于参数32B，精度FP16的模型，其需要的显存大概64G以上，若使用FP8量化，其显存占用降至32G，若采用INT8量化，显存占用更少。

| 问题     | 传统模型（FP32）   | 量化后模型（INT8/INT4）   |
| -------- | ------------------ | ------------------------- |
| 内存占用 | 巨大（7B模型28GB） | 大幅减少（7B模型约4-14G） |
| 推理速度 | 较慢               | 显著提升                  |
| 功耗     | 高                 | 低                        |
| 部署门槛 | 高（需要高端GPU）  | 低（可在消费级硬件运行）  |

## **2、量化的类型（PTQ和QAT）**

量化方法主要分为两类：训练后量化（Post-Training Quantization,PTQ）和量化感知训练（Quantization-Aware Training,QAT）。

**2.1训练后量化（PTQ）**

训练后量化是在模型训练完成之后进行量化，无需重新训练，这种方法简单快捷，但可能带来一定的精度损失。常见的PTQ方法：

**1）动态量化：**在推理过程中动态的在每次前向传递期间计算激活的最小值和最大值，以提供动态的每个张量比例因子，从而实现高精度，但此模式下推理性能会受到影响。

**2）静态量化：**使用代表性的校准数据集来进行量化，通常比动态量化高效，但需要校准数据集，静态量化可能会引入一些量化误差，从而影响模型的精度。

**2.2 量化感知训练（QAT）**

QAT在训练过程中模拟量化操作，让模型在训练时就适应量化的影响，通常能获得比PTQ更好的精度，但是耗时久、成本高，投入较大。

**3、常用的量化技术**

**GPTQ：**一种基于二阶信息的高精度训练后量化方法，逐层优化权重，特别适用于大语言模型。

**AWQ：**通过考虑激活值分布来指导权重量化，保护重要权重。

**SmoothQuant：**通过平滑激活值的异常值来改善量化效果，将量化难度从激活值转移到权重，使得权重量化和激活量化都更加稳定。

**4、量化位宽选择**

不同位宽代表不同的参数精度，也决定了显存的占用和计算的效率。

常用精度和显存需求对照关系：

| 精度 | 位宽 | 单参数大小 | 7B模型存储 | 32B模型存储 |
| ---- | ---- | ---------- | ---------- | ----------- |
| FP32 | 32   | 4 Bytes    | 28GB       | 128GB       |
| FP16 | 16   | 2 Bytes    | 14GB       | 64GB        |
| INT8 | 8    | 1 Bytes    | 7GB        | 32GB        |
| INT4 | 4    | 0.5 Bytes  | 3.5GB      | 16GB        |

不同位宽，显存减少和精度损失及适用场景：

| 位宽      | 显存减少   | 精度损失 | 适用场景         |
| --------- | ---------- | -------- | ---------------- |
| FP32      | 1x（基准） | 0%       | 训练、高精度推理 |
| FP16/BF16 | 2x         | <0.5%    | 训练、推理       |
| INT8      | 4x         | 1-2%     | 通用推理         |
| INT4      | 8x         | 2-5%     | 资源受限部署     |

**5、LLM Compressor量化工具集**

LLM Compressor是vLLM提供的一个专注于优化vllm部署的模型，即大模型压缩的工具，它提供一系列模型压缩技术，包括量化、剪枝、蒸馏等方法，压缩后的模型与vLLM一起部署，实现高达5倍的推理加速。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/nfqPeiaJ7nqmzicNVfUq76Jm41VWI0RJ2ccwEeTHTnh3HFotCtoiaM4RfUoVLUbXZ9MXDun4icxb3ffq3R7yBbO7eQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=8)

**它提供了一个全面的工具集：**

- 应用各种各样的压缩算法，包括权重和激活量化、修剪等
- 无缝集成Hugging Face Transformers, Models, and Datasets
- 使用与vLLM兼容的模型存储格式safetensors
- 通过加速支持大型模型的高性能压缩

**核心特点：**

- **权重和激活量化：**通过减少模型大小，提高基于通用和服务器的应用程序的推理性能，支持GPTQ,AWQ,SmoothQuant,RTN算法和INT W8A8, FP W8A8格式。
- **仅权重量化：**针对延迟敏感型应用，缩小模型尺寸，提高推理性能，支持GPTQ,AWQ,RTN算法和INT W4A16，INT W8A16格式。
- **权重剪枝：**减少模型大小，提高所有用例的推理性能，支持SparseGPT, Magnitude, Sparse Finetuning。

**6、使用LLM Compressor量化Qwen3-32B**

Qwen3-32B默认的参数精度是bfloat16，仅模型权重和激活需要占用64G的显存，这里使用4块L20卡进行量化，量化至INT8 W8A8。整个过程分为四步：

- **准备****模型权重文件**
- **准备校准数据**
- **应用量化**
- **准确性评估**



**实战环境信息：**

| 组件          | 配置                              |
| ------------- | --------------------------------- |
| GPU           | 4 ×  NVIDIA L20 48GB              |
| 网络          | 25Gb/s  以太网                    |
| CPU           | Intel(R)  Xeon(R) Gold 6430 128核 |
| 内存          | 512GB  DDR5                       |
| 存储          | 3.84T  NVMe SSD * 4  软raid5      |
| OS            | Ubuntu 22.04.5 LTS                |
| kernel        | 6.8.0-79-generic                  |
| NVIDIA        | 580.82.07                         |
| CUDA          | 13.0                              |
| llmcompressor | 0.8.1                             |
| docker        | 28.4.0                            |
| PyTorch       | 2.8.0                             |
| vllm          | v0.12.0                           |

**6.1 安装llm-compressor**

为了避免包冲突问题（特别是vllm、llmcompressor、torch、compressed-tensors这里因版本问题踩过坑），创建python虚拟环境，安装相关包和依赖：

```
python3 -m venv llm-compressor
source llm-compressor/bin/activate
#安装llmcompressor
pip install llmcompressor==0.8.1  -i https://mirrors.aliyun.com/pypi/simple
```

**6.2 下载模型权重文件**

我们使用命令行方式后台下载，从modelscope魔塔社区下载：

```
nohup modelscope download --model="Qwen/Qwen3-32B" --local_dir "/data/DeepSeek/Qwen3-32B" > Qwen3-32B_download_nohup.log 2>&1
```

**6.3 准备校验数据**

将激活量化为INT8时，需要样本数据来估计激活尺度，对于通用指令调整模型，可以使用ultrachat数据集。数据集可以从HF上下载，也可以提前下载好：

```
添加环境变量，以解决HF网站访问的问题：
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset HuggingFaceH4/ultrachat_200k --local-dir /data/DeepSeek/dataset/ultrachat_200k
```

**6.4 执行量化**

**1）加载模型**

首先使用标准transformers的AutoModelForCausalLM类加载模型和tokenizer:

```
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "/data/DeepSeek/Qwen3-32B" #本地模型权重文件位置
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto",)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

**2）加载并预处理校准数据集**

```
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512#样本的起点512（若准确性下降，可以增加）
MAX_SEQUENCE_LENGTH = 2048#序列长度
# 加载并预处理数据集
ds = load_dataset("/data/DeepSeek/dataset/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)
```

**3）应用量化**

当数据集加载好之后，就可以进行量化。对于量化算法的选择，主要考虑以下几点：

- 采用SmoothQuant使量化激活更容易一些
- 使用GPTQ将权重量化为8位
- 使用动态每令牌策略量化激活

注意这个过程会很长，我是4张L20 48G的卡，量化Qwen3-32B到INT8 W8A8耗时近5个小时（之前在H20上量化DeepSeek-R1-Distill-Qwen-32B到INT8，耗时近2小时）所以这里最好是在后台执行，执行完之后模型的权重和激活都被量化为INT8，且权重文件和tokenizer存储到模型文件所在目录。

```
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Configure the quantization algorithms to run.
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

由于整个过程耗时较久，可以将模型加载、数据集加载、模型量化放在一个脚本里后台执行：

```
nohup python3 qwen3-32b-Quantization-int8-w8a8.py > qwen3-32b-Quantization-int8-w8a8.log 2>&1
```

脚本内容：

```
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "/data/DeepSeek/Qwen3-32B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto",)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
ds = load_dataset("/data/DeepSeek/dataset/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Configure the quantization algorithms to run.
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

我们检查输出的模型文件，可以看到配置文件里有了量化的参数：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/nfqPeiaJ7nqmzicNVfUq76Jm41VWI0RJ2cpQaG6sjjI18CIXWlpRjux5PVnu2mwLe6sxoAmS3UzRsCQDdKkEz0tw/640?wx_fmt=png&from=appmsg&watermark=1&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=9)

**6.5 评估准确性**

以上步骤，我们已经得到了一个量化后的模型（INT8 W8A8），这里使用vllm加载模型，并使用lm_eval评估其准确性。

运行模型：

```
CUDA_VISIBLE_DEVICES='1'
from vllm import LLM
model = LLM("/data/DeepSeek/Qwen3-32B-W8A8-Dynamic-Per-Token",tensor_parallel_size=1,max_model_len=2048,max_num_seqs=2,gpu_memory_utilization=0.95)
```

可以看到模型在单张L20卡上加载成功：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/nfqPeiaJ7nqmzicNVfUq76Jm41VWI0RJ2ceDhLqnUUic9IIqlpEOM8ss4HeLmI3GufQuOjqWkn2fwTSGWKdH9j4Mw/640?wx_fmt=png&from=appmsg&watermark=1&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=10)

使用lm_eval评估模型，运行以下命令在GSM-8K上测试精度：

```
lm_eval --model vllm \
  --model_args pretrained="/data/DeepSeek/Qwen3-32B-W8A8-Dynamic-Per-Token",add_bos_token=true,dtype=auto,tensor_parallel_size=2,max_model_len=2048,max_num_seqs=2,gpu_memory_utilization=0.9 \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

最终会输出如下结果：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/nfqPeiaJ7nqmzicNVfUq76Jm41VWI0RJ2csDMdo8FxC8unVvEqs3emXx1nrJU47ygFMiawXc0UyyARvaZX9q2Q4Yw/640?wx_fmt=png&from=appmsg&watermark=1&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=11)

**7、量化遇到问题**

**7.1 校准数据集下载问题**

INT8量化需要准备校准数据，一般我们使用HF提供的数据集，会自动从HF上去下载数据集到本地：

```
# 加载并预处理数据集
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
```

这个过程会因HF访问而超时，可以提前设置环境变量，这是一个国内HF镜像网站：

```
export HF_ENDPOINT=https://hf-mirror.com
```

**7.2 GSM-8K数据集下载问题**

跟校准数据集一样，准确性评估阶段，我们使用了gsm8k，所以也会去HF下载数据，所以跟上面一样，设置国内HF镜像网站即可解决访问超时：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/nfqPeiaJ7nqmzicNVfUq76Jm41VWI0RJ2cvJO2ELYznpvXQcIdcmTJicfGAiaI4qE1GhqS2D4JoYVnOvJLFx3tTHHw/640?wx_fmt=png&from=appmsg&watermark=1&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=12)

设置环境变量后，数据集正常下载：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/nfqPeiaJ7nqmzicNVfUq76Jm41VWI0RJ2cI18HCV3nNclMv5ibsdjK3L1UvfxbhUMfcvgrFX9lVVapeWHdCSudmCg/640?wx_fmt=png&from=appmsg&watermark=1&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=13)

最终评估执行成功：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/nfqPeiaJ7nqmzicNVfUq76Jm41VWI0RJ2cic9iaP9Nib5UGiaM4KETo1BJK7picq8oaWx3FmWCeM8Ym4muicfxuwy35P9w/640?wx_fmt=png&from=appmsg&watermark=1&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=14)

**7.3 显存不足导致模型load失败**

模型量化之后，使用vllm加载模型

```
from vllm import LLM
model = LLM("/data/DeepSeek/Qwen3-32B-W8A8-Dynamic-Per-Token")
```

这里默认使用一块L20，然后报显存不足：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/nfqPeiaJ7nqmzicNVfUq76Jm41VWI0RJ2ciblQrnq4eWwXl7StNcWXTbb1SFabmVKoMfqgxHtAkWpnRrRxibia0mncQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=15)

最后设定了上下文长度、请求数、显存使用率提到95%，最终加载成功命令：

```
CUDA_VISIBLE_DEVICES='1'
from vllm import LLM
model = LLM("/data/DeepSeek/Qwen3-32B-W8A8-Dynamic-Per-Token",tensor_parallel_size=1,max_model_len=2048,max_num_seqs=2,gpu_memory_utilization=0.95)
```

**8 部署量化后Qwen3-32B**

这里采用容器化部署，在一块L20卡上运行模型推理服务，上下文长度2048，并发2，一块卡就能运行Qwen3-32B推理模型：

```
docker run -d \
--gpus all \
--restart always \
--name Qwen3-32B-Q \
--network host \
--shm-size 10.24g \
-e TZ=Asia/Shanghai \
-e GLOO_SOCKET_IFNAME=bond0 \
-e CUDA_VISIBLE_DEVICES='1' \
-v /data/DeepSeek/Qwen3-32B-W8A8-Dynamic-Per-Token:/model \
vllm/vllm-openai:v0.12.0 \
--model /model \
--dtype auto \
--api-key OPENWEBUI123 \
--max-num-seqs 2 \
--served-model-name Qwen3_32B \
--port 8000 \
--tensor-parallel-size 1 \
--max-model-len 2048
```

可以看到使用到GPU1，显存使用率86%左右：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/nfqPeiaJ7nqmzicNVfUq76Jm41VWI0RJ2cScMgweDhRmlcwPOJXqVna1leyniaSwq1HFhHiaZuuVG8MDgJiagcicCpWw/640?wx_fmt=png&from=appmsg&watermark=1&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=16)

**总结：**

本文分享了在大模型相关的一项优化技术-模型量化，它通过降低模型参数的精度，来压缩模型显存的占用和计算开销，使得模型能在受限的资源情况下部署和使用，大大降低了部署的门槛。vLLM提供了一个高效的量化工具LLM Compressor，我们使用这个工具对Qwen3-32B模型进行了INT8 W8A8的量化，其实像对于来源模型，huggingface和modelscope社区都提供了已经量化后的模型，直接下载即可，这里只是为了通过手动去量化一个模型，加深对模型量化概念和原理以及精度性能的理解。若大家有任何问题或者疑问，欢迎大家留言区讨论。
