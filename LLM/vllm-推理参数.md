## 推理参数
----

```bash
vllm serve /home/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --trust-remote-code \
  --tensor_parallel_size 4 \
  --port 8888 \
  --gpu-memory-utilization 0.8 \
  --max_num_seqs 32 \
  --max_model_len 56000 \
  --quantization sdaa_wint8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --max_num_batched_tokens 8192
```

| 参数                                                         | 含义                                | 备注 / 推荐值                                                |
| :----------------------------------------------------------- | :---------------------------------- | :----------------------------------------------------------- |
| `/home/Qwen3.8-27B`                                          | 模型本地路径                        | 需确保该路径下存在模型文件，且名称大小写与实际一致。         |
| `--served-model-name Qwen3.8-27B`                            | 对外暴露的模型名称                  | API 调用时的 `model` 参数需与此保持一致。                    |
| `--trust-remote-code`                                        | 允许加载模型仓库中的自定义代码      | 使用 Hugging Face 上包含自定义代码的非官方或新版模型时需要。 |
| `--tensor_parallel_size 4`                                   | 张量并行度（切分的 GPU 数量）       | 需要服务器有至少 4 张可用的显卡。                            |
| `--port 8888`                                                | 服务端口                            | 默认为 8000，此处指定为 8888。                               |
| `--gpu-memory-utilization 0.8`                               | GPU 显存利用率（KV Cache 权重占比） | 默认 0.9；0.8 适合多卡 TP，能降低显存 OOM 风险。             |
| `--max_num_seqs 32`                                          | 最大并发序列数                      | 建议保留此参数。                                             |
| `--max_model_len 56000`                                      | 模型最大上下文长度（输入+输出）     | 56000 长度会占用较多 KV Cache 显存，需留意显存是否充足。     |
| `--quantization sdaa_wint8`                                  | INT8 权重量化方式                   | 用于减小显存占用并提升吞吐，需确认当前环境和模型是否支持该量化格式。 |
| `--enable-auto-tool-choice`                                  | 启用自动工具选择                    | 适合带有工具调用能力的模型，让模型自行判断是否调用工具。     |
| `--tool-call-parser qwen3_coder`                             | 工具调用解析器                      | 专门用于解析 Qwen3 Coder 格式输出的工具调用结构。            |
| `--reasoning-parser qwen3`                                   | 推理解析器                          | 用于提取 Qwen3 模型的思维链（Reasoning Chain）或中间推理过程。 |
| `--enable-prefix-caching`                                    | 启用前缀缓存                        | 可显著降低重复系统提示词或长上下文的 KV Cache 重复计算。     |
| `--compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` | CUDA Graph 编译配置                 | 仅对 Decode 阶段启用 CUDA Graph 加速，降低首字延迟。         |
| `--max_num_batched_tokens 8192`                              | 单批次预填充最大 token 预算         | 决定了单次 Prefill 阶段的最大 token 处理量，建议根据显存情况调整。 |


## 推理流程

---

“单次 Prefill 阶段的最大 token 处理量”（即 `--max-num-batched-tokens 8192` 这个参数）。

在大模型推理中，整个过程分为两个阶段：

1. **Prefill（预填充）**：相当于把用户输入的一大段文字（Prompt）一次性送进模型去“阅读理解”。
2. **Decode（解码）**：阅读理解完后，模型一个字一个字地往外蹦。这个过程主要是消耗显存带宽，计算量相对较小。

`--max-num-batched-tokens 8192` **规定工厂里每次同时装车、并发处理的最大货物总件数（Token 总数）的上限**。

### **“单次 Prefill 阶段”的硬性限制**

因为在 Prefill 阶段，一个请求可能会包含成千上万个 Token（比如几千字的提问），它们被一次性打包送进 GPU 计算。

- **防止“爆仓”（显存 OOM）**：若这个上限设置得过大，当几个长文本请求同时涌进来时，一次性需要处理的 Token 就会暴增，直接撑爆 GPU 的显存，导致服务崩溃。
- **防止“霸占资源”**：若某个超级长的请求（比如 50K Token）直接塞进去，它会瞬间占满 GPU 的所有算力。此时，其他正在等待回复的短请求就会卡顿，必须等它处理完才能继续。设置 8192 的上限，就相当于把超长文本切成小块处理，避免它独占资源。

### **来实战推演：**

假设你的服务器正在同时处理 4 个并发请求，配额池上限是 **8192**：

- **场景 A（请求都短）**：
  请求 A（500 tokens）+ 请求 B（500 tokens）+ 请求 C（1000 tokens）+ 请求 D（800 tokens） = **2800 tokens**。
  因为 2800 < 8192，这 4 个请求可以**完美打包在一起**，一次性并行送进 GPU 进行 Prefill 处理，吞吐量极高。
- **场景 B（有长有短）**：
  请求 E（10,000 tokens）刚发进来，请求 F（500 tokens）也发进来。
  由于 10,000 > 8192，若系统开启了切块处理（Chunked Prefill），它会把请求 E 的文本**切成一块**（比如先切出 7500 tokens），剩下的部分放在下一轮处理。这样，单次 Prefill 处理的 Token 数被严格限制在了 8192 以内，既保证了不爆显存，又留出了空余配额让请求 F 能插队一起被处理。

**总结来说**：
`max-num-batched-tokens 8192` 就像一个**安全阀**。它确保了在任何一次并发计算中，所有用户的输入总和不会超过 8192 个 Token，从而让 GPU 既能满载工作，又不会因为瞬间负荷过大而崩溃。

## 切块处理
----

在 vLLM 里，开启 Chunked Prefill 主要靠两个参数配合：**`--enable-chunked-prefill`** 和 **`--max-num-batched-tokens`**。

### **1. 最简配置**

```bash
--enable-chunked-prefill \
--max-num-batched-tokens 8192
```

其中：
- **`--enable-chunked-prefill`**：开启分块预填充，允许把长 prompt 拆成多个 chunk 处理，并与 decode 请求混合调度。
- **`--max-num-batched-tokens 8192`**：控制每次调度迭代最多处理多少 token，这个值会直接决定 chunk 的“预算”。

### **2. 优化Qwen3.8-27B 推理配置**

```bash
vllm serve /home/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --trust-remote-code \
  --tensor_parallel_size 4 \
  --port 8888 \
  --gpu-memory-utilization 0.8 \
  --max_num_seqs 32 \
  --max_model_len 56000 \
  --quantization sdaa_wint8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --max-num-batched-tokens 8192
```

### **3. 参数推荐**

|                 目标                 |               推荐方向                |                             说明                             |
| :----------------------------------: | :-----------------------------------: | :----------------------------------------------------------: |
| 降低 TTFT、减少长 prompt 阻塞 decode |  `--max-num-batched-tokens 512-2048`  |        chunk 更小，decode 更容易插队，首 token 更稳。        |
|             提升并发吞吐             | `--max-num-batched-tokens 4096-8192`  | 单次能塞更多 prefill token，但长请求仍可能短暂影响 decode。  |
|             长上下文场景             |    保持 `--enable-chunked-prefill`    | 你的 `max_model_len=56000`，开启后长 prompt 不会一次性霸占 GPU。 |
|               避免 OOM               | 不要盲目调大 `max-num-batched-tokens` |            它越大，单次 prefill 的显存压力越高。             |

### **4. 需要注意**

- vLLM V1 引擎里，Chunked Prefill 有时默认就是开启的，但生产部署建议**显式写 `--enable-chunked-prefill`**，避免版本差异导致行为不一致。
- `--max-num-batched-tokens` 不建议设得过小，比如 64、128，否则长请求会被切得太碎，整体延迟反而变差。
- 若后续同时使用 `--enable-prefix-caching`，chunk 大小最好与 `block_size` 对齐，否则前缀缓存命中率可能下降。

简单说：**加 `--enable-chunked-prefill`，再用 `--max-num-batched-tokens` 控制每次处理多少 token，就是开启 Chunked Prefill 的标准做法。**


## 关闭 服务端的Thinking
---

### **推荐配置方式**

#### **1. 服务端默认关闭 Thinking**

若你希望生产环境默认不输出推理过程，可以在启动命令里加上：

```bash
--default-chat-template-kwargs '{"enable_thinking": false}'
```

此时可保留：

```bash
--reasoning-parser qwen3
```

这样即使个别请求仍然触发了 Thinking，vLLM 也能正确解析；

若完全不需要解析，也可以把 `--reasoning-parser qwen3` 去掉。

#### **2. 按请求动态关闭**

若服务端仍保留 `--reasoning-parser qwen3`，客户端可以通过 OpenAI 兼容接口单独关闭：

```json
{
  "model": "Qwen3.8-27B",
  "messages": [
    {"role": "user", "content": "解释一下什么是 Chunked Prefill"}
  ],
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

Python SDK 中通常写成：

```python
extra_body={
    "chat_template_kwargs": {"enable_thinking": False}
}
```

部分 vLLM 版本也支持直接写：

```python
extra_body={"enable_thinking": False}
```

### **结合你当前命令的建议**

你现在的启动参数里有：

```bash
--reasoning-parser qwen3
```

若希望 **默认关闭 Thinking**，建议改为：

```bash
vllm serve /home/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --trust-remote-code \
  --tensor_parallel_size 4 \
  --port 8888 \
  --gpu-memory-utilization 0.8 \
  --max_num_seqs 32 \
  --max_model_len 56000 \
  --quantization sdaa_wint8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --max_num_batched_tokens 8192 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

若后续某些 Agent、代码推理任务需要重新开启思考，客户端请求里传：

```json
"chat_template_kwargs": {"enable_thinking": true}
```
即可覆盖服务端默认值。
