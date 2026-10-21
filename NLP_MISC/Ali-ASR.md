# Ali ASR

----



阿里目前的语音识别（ASR）模型主要围绕 **Qwen-Audio 系列、Qwen3-ASR 系列、Fun-ASR 系列** 和较早期的 **Paraformer** 展开。

## **主要 ASR 模型系列**

| 系列                   | 代表模型                                                     | 特点                                                         |
| :--------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **Qwen-Audio-3.1-ASR** | `qwen-audio-3.1-asr-flash-streaming`、`qwen-audio-3.1-asr-flash-filetrans`、`qwen-audio-3.1-asr-flash` | 最新一代语音识别大模型，支持实时流式、文件转写和短音频识别，支持热词、Prompt 上下文、说话人分离等。 |
| **Qwen-Audio-3.0-ASR** | `qwen-audio-3.0-asr-flash-streaming`、`qwen-audio-3.0-asr-flash-filetrans`、`qwen-audio-3.0-asr-flash` | 上一代主力模型，支持实时流式与离线文件识别，多语种和方言覆盖较广。 |
| **Qwen3-ASR 系列**     | `qwen3-asr-flash-realtime`、`qwen3-asr-flash-filetrans`、`qwen3-asr-flash`、`Qwen3-ASR-1.7B`、`Qwen3-ASR-0.6B` | 基于 Qwen3-Omni 和 AuT 语音编码器，支持 52 种语种/方言，提供开源版本和 API 版本；0.6B 版本吞吐高，适合大规模异步处理。 |
| **Fun-ASR 系列**       | `fun-asr-realtime`、`fun-asr`、`fun-asr-mtl`、`fun-asr-flash-2026-06-15`、`fun-asr-flash-8k-realtime` | 阿里通义百聆系列，支持实时与非实时识别，支持多语种、方言、说话人分离、Prompt 上下文等。 |
| **Paraformer**         | `paraformer-realtime-v2`、`paraformer-v2`、`paraformer-8k-v2` | 较早一代 ASR 模型，支持实时与非实时识别，适合部分传统 ASR 场景，但官方建议优先迁移到 Qwen-Audio / Fun-ASR / Qwen3-ASR。 |

## **选型建议**

- **实时语音识别、会议字幕、语音助手**：优先看 `qwen-audio-3.1-asr-flash-streaming`、`fun-asr-realtime`、`qwen3-asr-flash-realtime`。
- **录音文件转写、播客、访谈**：优先看 `qwen-audio-3.1-asr-flash-filetrans`、`fun-asr`、`qwen3-asr-flash-filetrans`。
- **需要说话人分离**：优先看 `qwen-audio-3.1-asr-flash-filetrans`、`qwen-audio-3.0-asr-flash-filetrans`、`fun-asr`、`fun-asr-mtl`。
- **需要情感识别**：优先看 `qwen3-asr-flash-realtime`、`qwen3-asr-flash-filetrans`。
- **需要开源部署或高吞吐离线处理**：优先看 `Qwen3-ASR-1.7B`、`Qwen3-ASR-0.6B`。

注意：旧版 `qwen-audio-asr` 已下线，被 `qwen3-asr-flash` 等模型替代。 实际可用模型、地域支持和接口名称以阿里云百炼 / 千问 AI 平台当前文档为准。

------



# SenseVoice 

----



**SenseVoice 是阿里通义实验室 FunAudioLLM 团队开源的多语言语音理解基础模型，核心卖点是“一次推理同时输出识别文本 + 语种 + 情绪 + 音频事件”，其中 Small 版本推理速度极快。**

## **模型介绍**

| 版本                 | 架构与参数量                               | 特点                                                         |
| :------------------- | :----------------------------------------- | :----------------------------------------------------------- |
| **SenseVoice-Small** | encoder-only / non-autoregressive，约 234M | 10 秒音频约 70ms 出结果，速度快、延迟低，适合实时语音、端侧部署、低算力环境 |
| **SenseVoice-Large** | encoder-decoder / autoregressive，约 1587M | 支持更多语言、更高精度，支持时间戳和更完整的事件检测         |

### **核心能力**

- **ASR 自动语音识别**：支持中文、粤语、英语、日语、韩语等 50+ 语言，中文和粤语识别效果优于 Whisper。
- **LID 语种识别**：自动判断语音所属语种。
- **SER 语音情感识别**：识别高兴、悲伤、愤怒、中性等情绪。
- **AED 音频事件检测**：检测背景音乐、掌声、笑声、哭声、咳嗽、喷嚏等。
- **ITN 逆文本规范化**：将口语化数字转为阿拉伯数字。
- **富文本输出**：一次推理可同时得到“说了什么 + 什么语言 + 什么情绪 + 有什么声音事件”。

## **部署与生态**

- 属于 **FunASR** 生态，支持 Python、C++、Java、C# 等客户端。
- 支持 ONNX、LibTorch、GGUF 等导出格式，可在 GPU、CPU、昇腾 NPU 等环境部署。
- 提供 WebUI、Docker、流式 WebSocket 服务和微调脚本。
- 阿里云原 SenseVoice 录音文件识别服务已宣布下线，目前主要走开源部署或迁移到 Paraformer / Fun-ASR / 千问 ASR。

### **预训练模型获取方式**

**1. ModelScope 魔搭社区**
模型主页：`https://www.modelscope.cn/models/iic/SenseVoiceSmall`

下载方式

```bash
# 方式一：SDK 下载
from modelscope import snapshot_download
model_dir = snapshot_download('iic/SenseVoiceSmall')

# 方式二：Git 克隆
git clone https://www.modelscope.cn/iic/SenseVoiceSmall.git

# 方式三：命令行下载
modelscope download --model iic/SenseVoiceSmall --local_dir ./SenseVoiceSmall

# 方式四：直接下载权重文件
wget https://modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/model.pt
```

**2. Hugging Face**
可在 Hugging Face 搜索 `SenseVoiceSmall` 获取模型权重；另有 `SenseVoiceSmall-GGUF` 等量化版本，适合 CPU 离线运行。

**3. GitHub / FunASR 仓库**
训练、推理、微调代码和示例可在 GitHub 的 SenseVoice / FunASR 仓库获取。



## 运行需求

----



**SenseVoice-Small 推荐 GPU 显存 ≥2GB、系统内存 ≥8GB；SenseVoice-Large 推荐 GPU 显存 ≥8GB、系统内存 ≥16GB。** 具体要看是否量化、是否用 GPU、是否开并发。

### **SenseVoice-Small**

| 运行方式                 | 显存 / 内存要求            | 说明                                                     |
| :----------------------- | :------------------------- | :------------------------------------------------------- |
| **GPU + FP16 / PyTorch** | 显存约 **2GB** 起          | 官方资料和部署实践里通常按 2GB 显存估算                  |
| **GPU + ONNX INT8 量化** | 显存约 **0.5–1.2GB**       | T4 上约 660MB，RTX 3080 上约 1.2GB                       |
| **CPU / ONNX 推理**      | 内存约 **0.6–1.5GB**       | 量化后空闲约 300MB，处理中约 800MB–1.2GB，峰值可到 1.5GB |
| **系统内存建议**         | 至少 **4GB**，推荐 **8GB** | 处理长音频、并发或同时加载标点模型时建议更高             |

Small 版本是 Encoder-only、非自回归模型，参数量约 2.34 亿，适合低显存设备、实时流式、端侧或边缘部署。

### **SenseVoice-Large**

| 运行方式     | 显存 / 内存要求         | 说明                                                         |
| :----------- | :---------------------- | :----------------------------------------------------------- |
| **GPU 部署** | 显存建议 **8GB+**       | Large 为 Encoder-Decoder、自回归模型，参数量约 15.87 亿，显存占用明显更高 |
| **系统内存** | 建议 **16GB+**          | 官方部署示例中推荐 16GB 以上内存                             |
| **存储**     | 建议 **50GB+** 可用空间 | 包含模型、依赖、缓存和推理中间文件                           |

Large 版本支持更多语言、更高精度和更完整的事件/情感输出，但推理延迟也更高：例如 10 秒音频约 1.6 秒左右。

### **实际选型建议**

- **显存 ≤4GB 或只有 CPU**：优先跑 **SenseVoice-Small ONNX INT8**。
- **显存 4–6GB**：可以跑 Small 的 FP16/FP32 版本，或 Small 多并发。
- **显存 8GB**：Small 很宽裕；Large 单路推理通常可以跑，但并发空间有限。
- **显存 ≥16GB**：更适合跑 Large，或同时部署 Small + 标点模型 + VAD + 说话人分离等链路。



#  FunASR

-----



FunASR 不是某一个单独的“模型”，而是阿里通义实验室 / 达摩院语音实验室开源的 **语音识别工具包 + 模型体系**。它把语音识别、语音活动检测、标点恢复、说话人分离、情感识别、音频事件检测等能力封装成统一接口，开发者可以按场景组合使用。

## **FunASR 的核心能力**

| 能力                 | 说明                                                         |
| :------------------- | :----------------------------------------------------------- |
| **语音识别 ASR**     | 支持中文、英文、多语种、方言、中英混合、实时流式和离线文件转写。 |
| **语音活动检测 VAD** | 自动判断哪段音频有人说话，减少无效计算。                     |
| **标点恢复**         | 给识别结果自动加逗号、句号、问号等。                         |
| **说话人分离**       | 区分不同说话人，适合会议、访谈、客服录音。                   |
| **情感 / 事件识别**  | 部分模型支持识别情绪、掌声、笑声、背景音乐等。               |
| **逆文本规范化 ITN** | 把“二零二五年三月十二号”规范成“2025年3月12日”。              |
| **热词定制**         | 可通过热词增强专有名词、人名、产品名、术语的识别效果。       |
| **多部署形态**       | 支持 Python SDK、Docker、ONNX、LibTorch、GGUF、流式 WebSocket、OpenAI 兼容接口等。 |

### **主要模型与版本**

FunASR 生态里常见的模型有这些：

| 模型                                  | 定位                                                         | 适合场景                                      |
| :------------------------------------ | :----------------------------------------------------------- | :-------------------------------------------- |
| **Fun-ASR 1.5**                       | 端到端语音识别大模型，MoE 架构，支持 30 种语言、汉语七大方言及 20 余种地方口音。 | 多语种会议、方言识别、跨语言混合场景          |
| **Fun-ASR-Realtime**                  | 流式实时识别模型，首字延迟低，支持 30 种语言和 16 种中文方言。 | 直播字幕、会议实时转写、语音助手              |
| **Fun-ASR-Nano**                      | 轻量级 LLM 架构 ASR 模型，约 800M 参数，支持多语种和方言。   | 本地部署、低延迟实时识别、私有化部署          |
| **SenseVoice-Small**                  | 轻量多任务语音理解模型，约 234M 参数，支持 ASR、语种识别、情感识别、音频事件检测。 | 低资源设备、实时语音、需要情绪/事件信息的场景 |
| **Paraformer-zh / Paraformer-Large**  | 中文 ASR 主力模型，非自回归，速度快。                        | 中文离线转写、高精度转录、批量处理            |
| **FSMN-VAD / CT-Transformer / cam++** | 辅助模型，分别负责语音检测、标点恢复、说话人分离。           | 与 ASR 模型组合使用，形成完整转写链路         |

## **典型使用方式**

**1. 本地部署 / 开源使用**
适合需要数据不出域、私有化部署、二次开发或低成本推理的场景。可以通过 `funasr` 库加载模型，例如：

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512")
res = model.generate(input="example.wav")
print(res[0]["text"])
```

也可以组合 VAD、标点、说话人分离等模块，构建完整转写流程。

**2. 云端 API 使用**
如果不想自己维护模型，可以通过阿里云百炼平台调用 Fun-ASR 相关 API，适合快速接入会议转写、录音整理、客服质检等应用。

### **预训练模型获取渠道**

| 渠道                    | 特点                                         | 适合谁                 |
| :---------------------- | :------------------------------------------- | :--------------------- |
| **ModelScope 魔搭社区** | 官方主发布平台，国内下载通常更快。           | 国内开发者、优先推荐   |
| **Hugging Face**        | 国际社区，模型版本较多，可使用国内镜像加速。 | 海外环境或需要 HF 生态 |
| **GitHub**              | 获取代码、部署脚本、微调示例和版本说明。     | 需要源码和工程集成     |
| **阿里云百炼**          | 直接调用云端 API，无需下载模型。             | 不想自建服务的企业用户 |

ModelScope 上可以搜索 `FunASR`、`Fun-ASR-Nano`、`SenseVoiceSmall`、`paraformer-zh` 等关键词；Hugging Face 上可搜索 `funasr` 组织下的模型仓库。

### **简单选型建议**

- **中文离线高精度转写**：优先看 **Paraformer**。
- **多语种、方言、实时转写**：优先看 **Fun-ASR 1.5 / Fun-ASR-Realtime**。
- **轻量本地部署、低延迟**：优先看 **Fun-ASR-Nano**。
- **不仅要转写，还要语种、情绪、事件**：优先看 **SenseVoice-Small**。
- **需要说话人分离、标点、VAD**：搭配 **cam++、CT-Transformer、FSMN-VAD**。

阿里云上原 **SenseVoice 录音文件识别服务已宣布下线**，如果走云端服务，建议迁移到 Paraformer、Fun-ASR 或千问 ASR 系列。

------



# 2K3000 运行

----



针对**6.4 TOPS@INT8 + 2GB 显存 + 8GB 内存** 的本地算力，SenseVoice-Small 的 **INT8 ONNX 版本** 是最合适的起点。



## **优先获取路径**

| 渠道              | 推荐模型/关键词                                      | 说明                                         |
| :---------------- | :--------------------------------------------------- | :------------------------------------------- |
| **ModelScope**    | `iic/SenseVoiceSmall-onnx`                           | 官方 ONNX 版本，优先查                       |
| **ModelScope**    | `danieldong/sensevoice-small-onnx-quant`             | 社区常见的 INT8 量化 ONNX 版本               |
| **ModelScope**    | `manyeyes/sensevoice-small-onnx`                     | 另一个 ONNX 版本，适合 C#/多语言调用         |
| **Hugging Face**  | 搜索 `SenseVoiceSmall-onnx` / `SenseVoiceSmall GGUF` | 备选渠道                                     |
| **GitHub FunASR** | `onnx_models/sense_voice_small_int8`                 | 部分 FunASR 部署包里会附带导出后的 ONNX 模型 |

### **推荐先试这两个**

1. **ModelScope：`iic/SenseVoiceSmall-onnx`**
   这是官方 SenseVoice-Small 的 ONNX 版本仓库，通常比原始 `iic/SenseVoiceSmall` 更适合直接做 ONNX Runtime 推理。
2. **ModelScope：`danieldong/sensevoice-small-onnx-quant`**
   这是已经做过 INT8 量化的 ONNX 版本，体积更小，适合 2GB 显存、8GB 内存环境。

## **快速验证方案**

如果你只是想最快跑起来，可以直接用 `funasr-onnx`：

```bash
pip install funasr-onnx onnxruntime
```

推理示例：

```python
from funasr_onnx import SenseVoiceSmall

model = SenseVoiceSmall(
    model_dir="./sensevoice-small-onnx-quant",
    batch_size=1,
    quantize=True
)

result = model(["test.wav"], language="auto", use_itn=True)
print(result)
```

如果 NPU 支持 ONNX 模型，建议优先确认它是否支持 **ONNX Runtime 的 NPU EP / QNN EP / SNPE EP**；

如果只支持标准 ONNX，就先按 CPU/INT8 模式验证，再考虑把模型转换到 NPU 专用格式。

------



## 测试音频样例

----

测试音频不需要额外找，**官方仓库和 ModelScope 模型包里都自带了现成的示例音频**，而且阿里云 OSS 上还有一批可直接 `wget` 的公开测试 wav。

### **最省事的方式：用模型包自带的 example**

从 ModelScope 或 Hugging Face 下载 `iic/SenseVoiceSmall` 模型后，模型目录里自带 `example/` 文件夹，里面有官方示例音频：

- `example/en.mp3` — 英文示例
- `example/zh.mp3` — 中文示例

官方 `demo1.py` / `demo_onnx.py` 的默认输入路径就是 `{model_path}/example/en.mp3`，下载完模型直接跑 demo 就能验证，不用自己找文件。

### **官方公开测试音频直链（可直接下载）**

这些是官方 `train_wav.scp` 里登记的标准测试音频，都是 16kHz wav，覆盖中文、中英混合、英文：

- [asr_example_zh.wav（中文）](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav)
- [asr_example_cn_en.wav（中英混合）](https://xn--isv-ry6fm22j.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cn_en.wav)
- [asr_example_en.wav（英文）](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav)
- [BAC009S0764W0121.wav（中文，AISHELL 风格）](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav)
- [BAC009S0916W0489.wav（中文）](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav)

批量下载命令：

```bash
mkdir -p test_audio && cd test_audio
for f in asr_example_zh.wav asr_example_cn_en.wav asr_example_en.wav \
         BAC009S0764W0121.wav BAC009S0916W0489.wav; do
  curl -L "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/$f" -o "$f"
done
```

### **对应标注**

| 音频文件              | 语种标签 | 情感          | 事件         | 参考文本                                                   |
| :-------------------- | :------- | :------------ | :----------- | :--------------------------------------------------------- |
| BAC009S0764W0121.wav  | `<|zh|>` | `<|NEUTRAL|>` | `<|Speech|>` | 甚至出现交易几乎停滞的情况                                 |
| BAC009S0916W0489.wav  | `<|zh|>` | `<|NEUTRAL|>` | `<|Speech|>` | 湖北一公司以员工名义贷款数十员工负债千万                   |
| asr_example_cn_en.wav | `<|zh|>` | `<|NEUTRAL|>` | `<|Speech|>` | 所有只要处理 data 不管你是做 machine learning…（中英混合） |
| asr_example_en.wav    | `<|en|>` | `<|NEUTRAL|>` | `<|Speech|>` | he tried to think how it could be                          |

### **针对你 ONNX 验证场景的建议**

- **先验证 ONNX 推理链路**：用 `example/en.mp3` 或 `asr_example_zh.wav`，文件小、内容短，最容易定位问题是出在模型加载还是音频预处理上。
- **再验证多语种与 ITN**：用 `asr_example_cn_en.wav`，能同时检验中英混合识别和逆文本规范化。
- **输入格式注意**：SenseVoice 期望 **16kHz 单声道 PCM**。如果你的 NPU 推理框架不支持 mp3 解码，先把 mp3 转成 wav：`ffmpeg -i en.mp3 -ar 16000 -ac 1 en_16k.wav`。
- **验证富文本输出**：想测情感/事件标签，需要找带笑声、掌声的音频，官方示例音频基本都是 `<\|NEUTRAL\|> + <\|Speech\|>`，测不出事件检测能力。

