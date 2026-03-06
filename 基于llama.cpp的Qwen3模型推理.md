# llama.cpp 环境部署
---

## 基础环境说明
---

WSL2+Ubuntu24.04, CUDA版本信息如下：

```bash
(llama_inf) shi19@AIPC:~/llamaCPP/models$ nvidia-smi
Fri Mar  6 17:02:18 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.288.01             Driver Version: 581.83       CUDA Version: 13.0     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4080 ...    On  | 00000000:01:00.0  On |                  N/A |
| N/A   43C    P8               6W / 138W |    842MiB / 12282MiB |      2%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     50252      C   /llama-cli                                N/A      |
+---------------------------------------------------------------------------------------+
(llama_inf) shi19@AIPC:~/llamaCPP/models$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```

### llama.cpp构建与验证

1.安装依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl libcurl4-openssl-dev
```

2.克隆与构建

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# 创建构建目录
mkdir build && cd build

# 配置 CMake：启用 CUDA
# -DGGML_CUDA=ON 是关键参数
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local

# 开始编译 (利用你的 4080 多核优势，使用 -j 加速)
# 根据你的 CPU 核心数调整 -j 后面的数字，例如 -j$(nproc)
make -j$(nproc)
```

3.完成验证

```bash
(llama_inf) shi19@AIPC:~/llamaCPP/models$ llama-cli --version
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4080 Laptop GPU, compute capability 8.9, VMM: yes
version: 8211 (a0ed91a44)
built with GNU 13.3.0 for Linux x86_64
```
4.回到llama.cpp源码，安装Python绑定(可选)

注意：这里统一在miniconda虚拟环境中执行

```bash
# 回到 llama.cpp 根目录
cd ..

# 设置环境变量告诉 pip 启用 CUDA
export CMAKE_ARGS="-DGGML_CUDA=ON"
export FORCE_CMAKE=1

# 安装 python 包
pip install llama-cpp-python
```

## 模型验证
---

根据目前的公开信息和社区资源，Qwen3-1.7B 系列本身并没有官方直接发布的 GGUF 格式量化模型（即阿里巴巴通义实验室官方仓库中通常只提供 PyTorch/Safetensors 格式的原始权重或 FP8 版本）。

这里需要利用modelscope或HF上的官方模型，进行量化操作。

### 模型获取

下载方式：

```bash

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-1.7B')
```
或者
```
modelscope download --model Qwen/Qwen3-1.7B
```

保存位置：

```bash
(llama_inf) shi19@AIPC:~/llamaCPP/models$ ls ~/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B/
LICENSE    config.json         generation_config.json  model-00001-of-00002.safetensors  model.safetensors.index.json  tokenizer_config.json
README.md  configuration.json  merges.txt              model-00002-of-00002.safetensors  tokenizer.json                vocab.json
```

### 模型转换与量化

前置工作创建并激活虚拟环境
```bash
conda create -n vllm python=3.11 -y
conda activate llama_inf
```

```bash
python convert_hf_to_gguf.py ~/.cache/modelscope/hub/models/Qwen/Qwen3-1___7B/ --outfile ../models/qwen3-1.7b-f16.gguf
....
INFO:gguf.gguf_writer:Writing the following files:
INFO:gguf.gguf_writer:../models/qwen3-1.7b-f16.gguf: n_tensors = 311, total_size = 4.1G
Writing: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.06G/4.06G [00:23<00:00, 176Mbyte/s]
INFO:hf-to-gguf:Model successfully exported to ../models/qwen3-1.7b-f16.gguf
```
备注：转换脚本来自：llama.cpp源码

```bash
(llama_inf) shi19@AIPC:~/llamaCPP/models$ cd .. && llama-quantize  models/qwen3-1.7b-f16.gguf models/qwen3-1.7b-q4_k_m.gguf Q4_K_M

(llama_inf) shi19@AIPC:~/llamaCPP/models$ ls -al
total 5226732
drwxr-xr-x 3 shi19 shi19       4096 Mar  6 11:38 .
drwxr-xr-x 4 shi19 shi19       4096 Mar  6 10:55 ..
drwxr-xr-x 2 shi19 shi19       4096 Mar  6 11:14 ._____temp
-rw------- 1 shi19 shi19         92 Mar  6 11:14 .msc
-rw-r--r-- 1 shi19 shi19      13963 Mar  6 11:14 README.md
-rw-r--r-- 1 shi19 shi19       2803 Mar  6 11:13 model_download.py
-rw-r--r-- 1 shi19 shi19 4069679424 Mar  6 11:37 qwen3-1.7b-f16.gguf
-rw-r--r-- 1 shi19 shi19 1282439488 Mar  6 11:38 qwen3-1.7b-q4_k_m.gguf
```

### 量化模型推理

```bash
(llama_inf) shi19@AIPC:~/llamaCPP/models$ llama-cli -m ../models/qwen3-1.7b-q4_k_m.gguf -p "Hello, how are you?" -n 50 -ngl 999
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4080 Laptop GPU, compute capability 8.9, VMM: yes

Loading model...


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8211-a0ed91a44
model      : qwen3-1.7b-q4_k_m.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> Hello, how are you?

[Start thinking]
Okay, the user greeted me with "Hello, how are you?" I need to respond politely.

First, I should acknowledge their greeting. Maybe say "Hello!" to keep it friendly.

Then, I should offer a simple response. Since

[ Prompt: 83.0 t/s | Generation: 160.5 t/s ]

> /exit


Exiting...
llama_memory_breakdown_print: | memory breakdown [MiB]          | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - CUDA0 (RTX 4080 Laptop GPU) | 12281 = 5181 + (5831 =  1050 +    4480 +     300) +        1269 |
llama_memory_breakdown_print: |   - Host                        |                  254 =   166 +       0 +      88                |
```

## 扩展-推理参数ngl
---
### **1. 什么是** `-ngl` **(Number of GPU Layers)？**

LLM（大语言模型）的结构像是一个“千层蛋糕”，由很多个相同的 **Transformer 层（Block/Layer）** 堆叠而成。

- **Qwen3-1.7B** 大约有 **24~28 层**（具体取决于架构细节）。
- **Qwen-72B** 则有 **80 层**。

在推理时，数据必须**按顺序**一层一层地通过整个网络：

> 输入 -> 第1层 -> 第2层 -> ... -> 第N层 -> 输出

**`-ngl` 参数就是告诉 `llama.cpp`：把这个“蛋糕”的下面多少层搬到 GPU 显存里去计算。**

- **`-ngl 0`**：**CPU 模式**。所有层都在内存（RAM）中，用 CPU 计算。速度慢，但显存占用为 0。
- **`-ngl 999`**：**全 GPU 模式**。尝试把**所有层**都塞进 GPU 显存。只要显存够大，这就是**速度最快**的模式。
- **`-ngl 10`**：**混合模式**。前 10 层在 GPU 跑，剩下的层在 CPU 跑。

------

### **2. 为什么要设为 999？**

命令中设为 `999`，是一种**“贪心策略”**（尽可能多放）：

1. **避免手动数层数**：不同的模型层数不同（1.7B 是 20 多层，7B 是 30 多层，70B 是 80 层）。写死一个数字（比如 30）可能在 A 模型上够用，在 B 模型上就不够。
2. **自动截断**：`llama.cpp` 很聪明。如果你设了 `-ngl 999`，但模型总共只有 24 层，或者你的显存只够装 20 层，程序会**自动停止**在最大值（24 层或 20 层），而不会报错。
3. **追求极致速度**：GPU 的显存带宽（RTX 4080 高达 717 GB/s）远超系统内存带宽（通常 50-100 GB/s）。只要层在 GPU 上，计算速度就是飞快的；一旦有一层掉到 CPU，速度就会瞬间被 CPU 内存带宽卡脖子（可能从 50 tokens/s 跌到 5 tokens/s）。

**结论**：设 `999` 的意思是：“**请把能塞进显存的层全部塞进去，直到显存爆满为止，剩下的再给 CPU。**”

### **3. 图解：数据流向与瓶颈**

#### **情况 A：**`-ngl 999` **(且显存足够)**

```text
[GPU 显存]
| 第 1 层 | -> | 第 2 层 | -> ... -> | 第 N 层 |
   ⚡️高速   ⚡️高速             ⚡️高速
结果：速度极快 (RTX 4080 可达 50+ tokens/s)
```

*数据全程在 GPU 内部高速流转，无需经过 PCIe 总线。*

#### **情况 B：**`-ngl 10` **(显存不足，部分在 CPU)**

```text
[GPU 显存]                [系统内存 RAM]
| 第 1 层 | -> ... -> | 第 10 层 | --(PCIe 总线)--> | 第 11 层 | -> ...
   ⚡️高速                       🐢 瓶颈              🐢 慢速
```

*注意中间的箭头：每生成一个 token，数据都要从 GPU 拷到 CPU，算完再拷回来。PCIe 总线成了严重的瓶颈，导致整体速度大幅下降。*

------

### **4. 针对你的 RTX 4080 (16GB) 的实战建议**

你的显卡有 **16GB 显存**，这对于 1.7B 模型来说是**绰绰有余**的。

- **模型大小估算**：
  - Qwen3-1.7B (Q4_K_M 量化) 权重约 **1.3 GB**。
  - 上下文缓存 (KV Cache)：假设设置 4096 长度，约占 **0.5 GB**。
  - **总占用**：约 **2 GB**。
- **策略**：
  - 直接使用 **`-ngl 999`**。
  - 你的 16GB 显存可以轻松吃掉这 2GB，甚至还能同时跑好几个这样的模型，或者把上下文开到 32k/64k。
  - **不需要**担心显存溢出（OOM）。只有当你运行 14B 或 32B 的大模型且上下文很长时，才需要调整 `-ngl` 的值（例如设为 40 或 50）来防止爆显存。

### **5. 如何验证是否生效？**

运行命令时，观察启动日志：

```bash
./bin/llama-cli -m model.gguf -ngl 999 ...
```

**成功标志**：
日志中会出现类似这样的行：

```bash
llama_load_model_from_file: offloading 24 repeating layers to GPU
llama_load_model_from_file: offloaded 24/24 layers to GPU
```

- 如果显示 `offloaded 24/24`，说明**全部_layers_都在 GPU 上，完美！
- 如果显示 `offloaded 10/24`，说明显存不够，只有 10 层在 GPU，剩下 14 层在 CPU（此时速度会慢）。

### **总结**

- **含义**：`-ngl` 控制有多少层模型结构在 GPU 上运行。
- **为什么是 999**：这是一个“最大努力”值，意为“能放多少放多少”，让程序自动根据显存大小决定上限。
- **你的情况**：对于 1.7B 模型，放心使用 `-ngl 999`，这将发挥你 RTX 4080 的全部性能。

## 扩展量化常识
---

这组内容列出了 `llama.cpp` (GGUF 格式) 中针对 Qwen3-1.7B 模型常见的几种量化等级。它们的核心区别在于**模型文件大小**、**运行速度**（推理延迟）以及**智能程度**（精度/困惑度）。

以下是详细的对比分析，帮助你根据硬件条件做出选择：

### **1. 核心指标对比表**

| 量化等级    | 近似位宽 (bits) | 1.7B 模型大小 (约) | 显存/内存占用 |        精度损失         | 推理速度 | 推荐场景                     |
| :---------- | :-------------: | :----------------: | :-----------: | :---------------------: | :------: | :--------------------------- |
| **IQ2_XXS** |    ~2.0 bit     |      ~500 MB       |    < 1 GB     |  **高** (可能逻辑混乱)  |  ⚡️ 极快  | 极端受限设备 (如旧手机、MCU) |
| **Q4_K_M**  |    ~4.5 bit     |      ~1.1 GB       |    ~1.3 GB    | **极低** (几乎不可感知) |   ⚡️ 快   | **首选推荐** (平衡点)        |
| **Q5_K_M**  |    ~5.5 bit     |      ~1.3 GB       |    ~1.5 GB    |        微乎其微         |  🚀 较快  | 对精度有轻微洁癖的用户       |
| **Q6_K**    |    ~6.5 bit     |      ~1.5 GB       |    ~1.7 GB    |        几乎为零         |  🐢 中等  | 需要接近原始精度的场景       |
| **Q8_0**    |    ~8.0 bit     |      ~1.9 GB       |    ~2.1 GB    |   **无** (接近 FP16)    |  🐌 较慢  | 调试、基准测试、显存充足     |

> *注：模型大小为估算值，具体取决于 tokenizer 大小和元数据。1.7B 模型本身很小，因此不同量化等级间的体积差异绝对值不大（几百 MB），但在低配设备上这几百 MB 可能是能否运行的关键。*

### **2. 详细解读与选择建议**

#### **🏆** ***\*Q4_K_M (黄金标准)\****

- **特点**：这是目前社区公认的“甜点”配置。它使用了混合精度量化策略（部分权重用 4-bit，部分用 6-bit），在大幅减小体积的同时，最大程度保留了模型的逻辑推理能力。
- **为什么推荐**：对于 1.7B 这样的小模型，参数量本来就少，如果量化过度（如 2-bit），模型容易“变傻”（胡言乱语或逻辑断裂）。Q4_K_M 能在 1.1GB 的体积下，提供接近原始模型 98%-99% 的性能。
- **适用**：90% 的用户，包括笔记本电脑、树莓派、中低端手机。

#### **🥈** ***\*Q5_K_M / Q6_K (精度优先)\****

- **特点**：随着位宽增加，模型越来越接近原始未量化版本（FP16）。
- **差异**：在 1.7B 这种小模型上，Q5 和 Q6 相比 Q4 的提升可能不如在大模型（如 70B）上那么明显。除非你发现 Q4 版本在处理复杂数学或代码时偶尔出错，否则没必要为了这点精度牺牲速度和增加显存占用。
- **适用**：显存充裕（>4GB），且对输出稳定性要求极高的场景。

#### **🥉** ***\*Q8_0 (无损/近无损)\****

- **特点**：8-bit 量化通常被视为“无损”，其效果与原始的 16-bit 模型几乎没有肉眼可见的区别。
- **缺点**：文件体积几乎是 Q4 的两倍，推理速度也慢得多（因为读取的数据量大了，且计算优化不如低比特好）。
- **适用**：主要用于开发者测试模型底线，或者作为基准来对比其他量化版本的损失情况。日常使用性价比低。

#### **⚠️** ***\*IQ2_XXS (极限压缩)\****

- **特点**：这是 `llama.cpp` 引入的新型超低位宽量化（Importance Matrix Quantization）。它试图在 2-bit 下保留尽可能多的信息。

- **风险**：对于 1.7B 这种**小参数模型**，2-bit 是非常危险的。模型可能会丧失基本的指令遵循能力，出现重复说话、逻辑崩塌或完全无法理解问题。

- **适用**：仅在硬件极其受限（例如只有 512MB 内存可用）且愿意牺牲智能程度来换取“能跑就行”的情况下使用。**一般用户不建议尝试。**
  
### **3. 最终建议**

针对 **Qwen3-1.7B** 这个具体的模型：

1. 首选下载：`Q4_K_M`。
   - 理由：1.7B 模型本身很小，Q4 版本在大多数现代设备（甚至几年前的手机）上都能流畅运行，且智力保留最完整。
2. **备选方案**：如果你的设备内存非常紧张（例如只剩 800MB 可用），可以尝试 **`IQ2_XXS`** 或寻找 **`Q3_K_M`**（如果有的话），但要做好模型变笨的心理准备。
3. 无需考虑：`Q8_0`。
   - 理由：对于 1.7B 模型，多占用的几百 MB 显存换来的性能提升在日常对话中几乎感觉不到，不如直接跑 Q4 获得更快的响应速度。

**一句话总结**：直接下载 **`...Q4_K_M.gguf`** 版本，它是速度、体积和智能程度的最佳平衡点。

`llama.cpp` 项目中定义的**量化方案（Quantization Schemes）**的代码名称。它们不仅仅代表“压缩了多少”，还代表了**如何压缩**（即哪些部分被压缩得更厉害，哪些部分保留了更高精度）。

我们可以把这些代码拆解为三个部分来理解：

1. **基础类型** (Q, IQ)
2. **策略等级** (K, S, M, L)
3. **具体数值/变体** (数字, XXS, XS)

------

### **1. 前缀：基础量化类型**

#### ***\*Q (Quantized)\****

- **含义**：标准的线性量化或混合精度量化。
- **特点**：这是最传统、最成熟的量化方式。它将权重矩阵均匀地或非均匀地映射到低比特整数（如 4-bit, 5-bit, 8-bit）。
- **适用**：大多数通用场景，稳定性好。

#### ***\*IQ (Importance Matrix Quantization)\****

- **含义**：基于**重要性矩阵**的量化（由 `llama.cpp` 开发者 `ggerganov` 引入的高级技术）。

- 核心逻辑

  ：它不是对所有权重一视同仁。它通过分析模型，找出哪些权重对输出结果影响最大（重要），哪些影响较小（不重要）。

  - **重要权重**：用更高的精度存储（甚至不量化）。
  - **不重要权重**：用极低的精度（如 1-bit 或 2-bit）甚至丢弃。

- **特点**：能在**极低比特率**（如 2-bit, 3-bit）下，比传统 Q 方法保留更多的模型“智力”。

- **适用**：极度压缩场景（如 `IQ2`, `IQ3`），试图在模型变很小的同时不让它变太傻。

------

### **2. 后缀：策略等级 (K, S, M, L)**

这些字母主要出现在 **K-quants**（一种改进的量化方法，将权重分块处理）中，代表**块内精度的分配策略**。

- **S (Small / Simple)**
  - **含义**：小版本。
  - **策略**：使用较少的辅助数据（scale/bias），压缩率最高，但精度损失相对较大。
  - **例子**：`Q4_K_S` 比 `Q4_K_M` 更小，但稍微笨一点。
- **M (Medium)**
  - **含义**：中等版本（**推荐默认值**）。
  - **策略**：平衡了大小和精度。它对部分权重块使用高精度，部分使用低精度，是一个经过调优的“甜点”。
  - **例子**：`Q4_K_M`, `Q5_K_M`。
- **L (Large)**
  - **含义**：大版本。
  - **策略**：使用更多的辅助数据来保留精度，文件更大，接近未量化版本的表现。
  - **例子**：`Q6_K` (实际上 Q6_K 通常只有一种策略，但在 Q5/Q4 中有 L 版本，如 `Q4_K_L` 较少见，通常 M 就足够了)。

> **注意**：对于 `Q2`, `Q3` 等低比特，通常只有 `S` 或特定变体，因为空间太紧张，无法做复杂的 M/L 区分。

------

### **3. 特殊变体：XXS, XS**

这些主要出现在 **IQ (重要性矩阵量化)** 系列中，表示**压缩的极端程度**。

- **XXS (Extra Extra Small)**
  - **含义**：超超小。
  - **特点**：极致的压缩。例如 `IQ2_XXS` 试图将平均位宽压到 2.0 bit 左右。它会极其激进地丢弃不重要的信息。
  - **风险**：模型可能会严重降智，出现逻辑混乱。
- **XS (Extra Small)**
  - **含义**：超小。
  - **特点**：比 XXS 稍微好一点点，比如 `IQ2_XS` 或 `IQ3_XS`。它在 2-bit 到 3-bit 之间寻找更好的平衡，保留稍微多一点的“重要权重”。
- **S / M / L (在 IQ 中)**
  - IQ 系列也有 `IQ2_S`, `IQ3_M` 等，逻辑同上：S 更压缩，M 更平衡，L 精度更高。

------

### **举例拆解**

#### ***\*案例 A:\** `Q4_K_M`**

- **Q4**: 基础目标是 4-bit 量化。
- **K**: 使用 K-quants 分块技术（比旧版 Q4_0/Q4_1 更聪明）。
- **M**: Medium 策略。在这个 4-bit 的框架下，它会把一些块量化为 4-bit，另一些关键块量化为 5-bit 或 6-bit，并保留适量的浮点缩放因子。
- **总结**：这是一个**平衡型**的 4-bit 模型。

#### ***\*案例 B:\** `IQ2_XXS`**

- **IQ**: 使用重要性矩阵技术，区分权重的重要性。
- **2**: 目标平均位宽约为 2-bit。
- **XXS**: Extra Extra Small 策略。这是该系列中**最激进**的压缩方案，只保留绝对核心的权重信息，其余全部大幅压缩。
- **总结**：这是一个**极限压缩型**模型，体积最小，但“智商”风险最大。

#### ***\*案例 C:\** `Q8_0`**

- **Q8**: 8-bit 量化。
- **0**: 这里的 `0` 是旧版命名习惯（相对于 `Q8_1`），表示没有额外的 per-block 缩放因子优化，或者说是标准的 8-bit 线性量化。
- **总结**：几乎无损，但体积大。

### **一张图看懂选择逻辑**



| 如果你想要...                 | 应该看哪个前缀/后缀？                | 典型代号            |
| :---------------------------- | :----------------------------------- | :------------------ |
| **最佳平衡 (日常用)**         | **Q** + **K** + **M**                | `Q4_K_M`, `Q5_K_M`  |
| **极致省空间 (能跑就行)**     | **IQ** + **XXS/XS**                  | `IQ2_XXS`, `IQ3_XS` |
| **稍微省点空间 (比平衡更小)** | **Q** + **K** + **S**                | `Q4_K_S`            |
| **最高精度 (不管体积)**       | **Q** + **高数字** 或 **IQ** + **L** | `Q8_0`, `IQ4_XL`    |

**简单记忆法：**

- **Q** = 标准款。
- **IQ** = 智能款（专门对付超低比特）。
- **S/M/L** = 小号/中号/大号（在同一种量化位数下，选 M 通常没错）。
- **XXS/XS** = 压缩狂魔（仅限 IQ 系列，小心模型变傻）。
