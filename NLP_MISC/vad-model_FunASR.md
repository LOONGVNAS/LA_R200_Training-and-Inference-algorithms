## VAD模型

---

**vad_model 是语音活动检测(Voice Activity Detection)模型，专门负责把连续音频切成“有效语音段”和“静音/噪声段”，本身不生成文本，也不负责加标点。**

### **vad_model 的作用**

在语音识别链路里，它通常做这几件事：

- **检测人声起止点**：判断音频里哪一段是人在说话。
- **切掉静音和噪声**：避免把空调声、键盘声、空白段送入 ASR。
- **把长音频切成短片段**：长音频直接识别容易显存/内存溢出，VAD 切段后逐段识别更稳。
- **辅助断句**：VAD 切出来的片段边界，会影响 ASR 输出的句子边界，但它不是标点模型。

FunASR 里常用的 VAD 模型是 **FSMN-VAD**，例如 `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch`。

### **ONNX 推理没有标点**

FunASR 的完整语音识别链路通常是三段式：

```
原始音频
  ↓ VAD 模型：切语音段
  ↓ ASR 模型：语音转文字
  ↓ 标点模型：加逗号、句号等
最终带标点文本
```

若你只用 ONNX 跑 ASR 模型，没有接 VAD 和标点模型，输出自然就是**连续文本、没有标点**。

### **建议的 ONNX 组合**

要获得接近 FunASR 默认体验，可以这样组合：

| 模块     | 作用             | 示例模型                          |
| :------- | :--------------- | :-------------------------------- |
| **VAD**  | 切语音段、去静音 | `fsmn-vad`                        |
| **ASR**  | 语音转文本       | `SenseVoiceSmall` / `Paraformer`  |
| **PUNC** | 加标点           | `ct-punc` / `punc_ct-transformer` |

FunASR 的 ONNX 链路里，VAD 和标点模型都可以单独导出为 ONNX，所以可以做成 **VAD → ASR → PUNC** 三段推理。



## 用FunASR 串联 VAD 和  ONNX

---



**用 funasr_onnx 把 VAD、ASR、标点模型串起来**，效果接近 FunASR 默认链路。

### **1. 安装依赖**

```bash
pip install -U funasr_onnx soundfile
```

### **2. Python 串联代码**

```python
import soundfile as sf
from pathlib import Path
from funasr_onnx import Fsmn_vad, SenseVoiceSmall, CT_Transformer_VadRealtime
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess


class ONNXFileASR:
    def __init__(
        self,
        vad_model_dir="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        asr_model_dir="iic/SenseVoiceSmall",
        punc_model_dir="damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    ):
        print("加载 VAD 模型 ...")
        self.vad_model = Fsmn_vad(vad_model_dir)

        print("加载 ASR 模型 ...")
        self.asr_model = SenseVoiceSmall(
            asr_model_dir,
            batch_size=10,
            quantize=True,
        )

        print("加载标点模型 ...")
        self.punc_model = CT_Transformer_VadRealtime(punc_model_dir)

    def process(self, wav_path: str) -> str:
        wav_path = Path(wav_path)
        audio, sr = sf.read(wav_path)

        # 1. VAD：切出有效语音段
        vad_res = self.vad_model(str(wav_path))
        segments = vad_res[0]  # [[start_ms, end_ms], ...]

        texts = []
        for start_ms, end_ms in segments:
            s = int(start_ms * sr / 1000)
            e = int(end_ms * sr / 1000)
            seg = audio[s:e]

            tmp = f".tmp_seg_{start_ms}_{end_ms}.wav"
            sf.write(tmp, seg, sr)

            # 2. ASR：语音转文本
            res = self.asr_model([tmp], language="auto", use_itn=True)
            text = rich_transcription_postprocess(res[0]) if res else ""

            # 3. 标点：给文本加标点
            if text:
                text = self.punc_model(text)[0]

            if text:
                texts.append(text)

        return " ".join(texts)


if __name__ == "__main__":
    asr = ONNXFileASR()
    result = asr.process("asr_example.wav")
    print(result)
```

### **3. 链路说明**

```vad
原始音频
  ↓ Fsmn_vad：切出有效语音段
  ↓ SenseVoiceSmall：ASR 识别
  ↓ CT_Transformer_VadRealtime：加标点
最终带标点文本
```

**4. 几个注意点**

- 如果你用的是 **Paraformer**，可以把 `SenseVoiceSmall` 换成：

  ```python
  from funasr_onnx import Paraformer
  self.asr_model = Paraformer(asr_model_dir, batch_size=10, quantize=True)
  ```

- 标点模型建议用 **`CT_Transformer_VadRealtime`**，它更适合分段文本加标点。

- 如果你的音频已经是短片段、没有明显静音，可以先不接 VAD，只跑 **ASR + 标点**。

- 如果想进一步提速，可以保留 `quantize=True`；如果遇到识别质量下降，再改成 `False`。

这套代码基本能把你说的“ONNX 推理更稳 + 标点缺失”两个问题一起解决。
