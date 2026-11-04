# 模型下载与转换

----

```bash
modelscope download --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 --local_dir ./Qwen2.5-14B-Instruct-GPTQ-Int8
```



## 模型转换-GPTQ->FP16-INT8

利用**HuggingFace / GPTQ 量化权重目录**，借助 llama.cpp 的 `convert_hf_to_gguf.py` 转成 GGUF，再根据需要选择 `q8_0`



先转成 F16 GGUF（中间文件）25.9GB

```bash
(llama_inf) shi19@AIPC:~/LLM_Exec$ python3 /home/shi19/llamaCPP/llama.cpp/convert_hf_to_gguf.py  ./Qwen2.5-14B-Instruct-GPTQ-Int8/     --outtype f16     --outfile qwen2.5-14b-instruct-f16.gguf
INFO:hf-to-gguf:Loading model: Qwen2.5-14B-Instruct-GPTQ-Int8
INFO:hf-to-gguf:Model architecture: Qwen2ForCausalLM
INFO:hf-to-gguf:gguf: loading model weight map from 'model.safetensors.index.json'
INFO:hf-to-gguf:gguf: indexing model part 'model-00001-of-00005.safetensors'
INFO:hf-to-gguf:gguf: indexing model part 'model-00002-of-00005.safetensors'
INFO:hf-to-gguf:gguf: indexing model part 'model-00003-of-00005.safetensors'
INFO:hf-to-gguf:gguf: indexing model part 'model-00004-of-00005.safetensors'
INFO:hf-to-gguf:gguf: indexing model part 'model-00005-of-00005.safetensors'
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:token_embd.weight,         torch.float16 --> F16, shape = {5120, 152064}
INFO:hf-to-gguf:blk.0.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.0.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.0.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.0.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.0.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.1.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.1.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.1.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.1.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.1.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.2.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.2.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.2.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.2.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.2.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.3.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.3.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.3.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.3.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.3.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.4.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.4.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.4.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.4.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.4.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.5.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.5.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.5.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.5.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.5.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.6.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.6.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.6.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.6.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.6.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.7.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.7.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.7.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.7.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.7.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.8.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.8.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.8.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.10.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.10.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.10.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.10.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.10.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.11.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.11.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.11.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.11.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.11.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.12.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.12.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.12.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.12.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.12.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.13.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.13.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.13.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.13.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.13.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.14.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.14.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.14.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.14.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.14.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.15.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.15.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.15.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.15.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.15.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.16.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.16.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.16.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.16.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.16.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.17.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.17.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.17.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.17.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.17.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.18.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.18.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.18.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.18.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.18.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.19.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.19.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.19.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.19.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.19.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.20.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.20.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.20.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.20.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.20.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.21.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.21.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.21.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.21.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.21.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.22.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.22.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.22.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.8.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.8.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.9.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.9.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.9.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.9.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.9.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.22.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.22.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.23.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.23.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.23.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.23.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.23.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.24.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.24.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.24.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.24.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.24.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.25.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.25.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.25.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.25.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.25.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.26.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.26.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.26.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.26.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.26.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.27.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.27.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.27.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.27.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.27.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.28.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.28.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.28.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.28.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.28.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.29.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.29.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.29.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.29.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.29.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.30.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.30.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.30.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.30.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.30.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.31.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.31.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.31.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.31.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.31.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.32.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.32.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.32.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.32.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.32.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.33.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.33.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.33.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.33.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.33.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.34.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.34.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.34.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.34.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.34.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.35.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.35.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.35.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.35.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.35.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.36.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.36.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.36.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.36.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.36.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.37.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.37.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.37.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.37.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.37.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.38.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.38.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.38.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.38.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.38.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.39.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.39.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.39.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.39.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.39.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.40.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.40.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.40.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.40.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.40.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.41.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.41.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.41.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.41.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.41.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.42.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.42.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.42.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.42.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.42.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.43.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.43.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.43.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.43.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.43.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.44.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.44.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.44.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.44.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.44.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.45.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.45.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.45.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.45.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.45.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.46.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.46.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.46.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.46.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.46.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.47.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.47.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.47.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.47.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.47.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:output_norm.weight,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:output.weight,             torch.float16 --> F16, shape = {5120, 152064}
INFO:hf-to-gguf:blk.0.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.0.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.0.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.0.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.0.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.0.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.0.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.1.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.1.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.1.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.1.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.1.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.1.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.1.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.2.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.2.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.2.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.2.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.2.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.2.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.2.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.3.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.3.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.3.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.3.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.3.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.3.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.3.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.4.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.4.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.4.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.4.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.4.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.4.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.4.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.5.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.5.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.5.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.5.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.5.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.5.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.5.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.6.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.6.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.6.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.6.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.6.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.6.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.6.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.7.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.7.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.7.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.7.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.7.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.7.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.7.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.8.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.8.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.8.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.8.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.8.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.10.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.10.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.10.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.10.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.10.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.10.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.10.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.11.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.11.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.11.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.11.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.11.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.11.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.11.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.12.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.12.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.12.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.12.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.12.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.12.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.12.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.13.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.13.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.13.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.13.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.13.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.13.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.13.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.14.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.14.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.14.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.14.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.14.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.14.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.14.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.15.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.15.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.15.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.15.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.15.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.15.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.15.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.16.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.16.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.16.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.16.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.16.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.16.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.16.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.17.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.17.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.17.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.17.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.17.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.17.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.17.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.18.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.18.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.18.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.18.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.18.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.18.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.18.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.19.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.19.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.19.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.19.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.19.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.19.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.19.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.20.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.20.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.20.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.20.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.20.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.20.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.20.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.21.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.21.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.21.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.21.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.21.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.21.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.21.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.22.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.22.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.22.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.22.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.22.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.8.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.8.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.9.ffn_down.weight,     torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.9.ffn_gate.weight,     torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.9.ffn_up.weight,       torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.9.attn_k.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.9.attn_output.weight,  torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.9.attn_q.weight,       torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.9.attn_v.weight,       torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.22.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.22.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.23.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.23.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.23.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.23.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.23.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.23.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.23.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.24.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.24.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.24.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.24.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.24.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.24.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.24.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.25.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.25.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.25.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.25.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.25.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.25.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.25.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.26.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.26.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.26.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.26.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.26.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.26.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.26.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.27.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.27.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.27.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.27.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.27.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.27.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.27.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.28.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.28.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.28.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.28.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.28.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.28.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.28.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.29.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.29.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.29.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.29.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.29.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.29.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.29.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.30.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.30.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.30.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.30.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.30.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.30.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.30.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.31.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.31.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.31.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.31.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.31.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.31.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.31.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.32.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.32.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.32.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.32.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.32.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.32.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.32.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.33.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.33.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.33.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.33.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.33.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.33.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.33.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.34.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.34.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.34.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.34.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.34.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.34.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.34.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.35.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.35.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.35.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.35.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.35.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.35.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.35.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.36.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.36.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.36.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.36.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.36.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.36.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.36.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.37.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.37.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.37.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.37.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.37.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.37.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.37.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.38.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.38.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.38.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.38.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.38.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.38.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.38.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.39.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.39.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.39.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.39.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.39.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.39.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.39.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.40.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.40.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.40.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.40.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.40.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.40.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.40.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.41.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.41.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.41.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.41.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.41.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.41.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.41.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.42.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.42.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.42.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.42.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.42.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.42.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.42.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.43.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.43.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.43.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.43.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.43.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.43.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.43.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.44.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.44.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.44.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.44.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.44.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.44.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.44.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.45.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.45.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.45.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.45.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.45.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.45.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.45.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.46.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.46.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.46.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.46.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.46.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.46.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.46.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.47.ffn_down.weight,    torch.float32 --> F16, shape = {13824, 5120}
INFO:hf-to-gguf:blk.47.ffn_gate.weight,    torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.47.ffn_up.weight,      torch.float32 --> F16, shape = {5120, 13824}
INFO:hf-to-gguf:blk.47.attn_k.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:blk.47.attn_output.weight, torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.47.attn_q.weight,      torch.float32 --> F16, shape = {5120, 5120}
INFO:hf-to-gguf:blk.47.attn_v.weight,      torch.float32 --> F16, shape = {5120, 1024}
INFO:hf-to-gguf:Set meta model
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:gguf: context length = 32768
INFO:hf-to-gguf:gguf: embedding length = 5120
INFO:hf-to-gguf:gguf: feed forward length = 13824
INFO:hf-to-gguf:gguf: head count = 40
INFO:hf-to-gguf:gguf: key-value head count = 8
WARNING:hf-to-gguf:Unknown RoPE type: default
INFO:hf-to-gguf:gguf: rope scaling type = NONE
INFO:hf-to-gguf:gguf: rope theta = 1000000.0
INFO:hf-to-gguf:gguf: rms norm epsilon = 1e-06
INFO:hf-to-gguf:gguf: file type = 1
INFO:hf-to-gguf:Set model quantization version
INFO:hf-to-gguf:Set model tokenizer
INFO:gguf.vocab:Adding 151387 merge(s).
INFO:gguf.vocab:Setting special token type eos to 151645
INFO:gguf.vocab:Setting special token type pad to 151643
INFO:gguf.vocab:Setting special token type bos to 151643
INFO:gguf.vocab:Setting add_bos_token to False
INFO:gguf.vocab:Setting chat_template to {%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}

INFO:gguf.gguf_writer:Writing the following files:
INFO:gguf.gguf_writer:qwen2.5-14b-instruct-f16.gguf: n_tensors = 579, total_size = 29.5G
Writing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29.5G/29.5G [08:23<00:00, 58.6Mbyte/s]
INFO:hf-to-gguf:Model successfully exported to qwen2.5-14b-instruct-f16.gguf
```



## 模型转换-GPTQ->IN8

----

````bash
(llama_inf) shi19@AIPC:~/LLM_Exec$  python3 /home/shi19/llamaCPP/llama.cpp/convert_hf_to_gguf.py  ./Qwen2.5-14B-Instruct-GPTQ-Int8/     --outtype q8_0 \
    --outfile qwen2.5-14b-instruct-q8_0.gguf
INFO:hf-to-gguf:Loading model: Qwen2.5-14B-Instruct-GPTQ-Int8
INFO:hf-to-gguf:Model architecture: Qwen2ForCausalLM
INFO:hf-to-gguf:gguf: loading model weight map from 'model.safetensors.index.json'
INFO:hf-to-gguf:gguf: indexing model part 'model-00001-of-00005.safetensors'
INFO:hf-to-gguf:gguf: indexing model part 'model-00002-of-00005.safetensors'
INFO:hf-to-gguf:gguf: indexing model part 'model-00003-of-00005.safetensors'
INFO:hf-to-gguf:gguf: indexing model part 'model-00004-of-00005.safetensors'
INFO:hf-to-gguf:gguf: indexing model part 'model-00005-of-00005.safetensors'
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:token_embd.weight,         torch.float16 --> Q8_0, shape = {5120, 152064}
INFO:hf-to-gguf:blk.0.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.0.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.0.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.0.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.0.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.1.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.1.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.1.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.1.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.1.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.2.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.2.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.2.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.2.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.2.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.3.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.3.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.3.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.3.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.3.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.4.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.4.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.4.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.4.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.4.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.5.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.5.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.5.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.5.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.5.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.6.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.6.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.6.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.6.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.6.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.7.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.7.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.7.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.7.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.7.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.8.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.8.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.8.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.10.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.10.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.10.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.10.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.10.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.11.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.11.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.11.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.11.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.11.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.12.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.12.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.12.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.12.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.12.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.13.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.13.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.13.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.13.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.13.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.14.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.14.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.14.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.14.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.14.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.15.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.15.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.15.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.15.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.15.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.16.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.16.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.16.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.16.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.16.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.17.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.17.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.17.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.17.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.17.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.18.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.18.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.18.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.18.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.18.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.19.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.19.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.19.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.19.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.19.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.20.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.20.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.20.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.20.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.20.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.21.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.21.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.21.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.21.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.21.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.22.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.22.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.22.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.8.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.8.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.9.attn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.9.ffn_norm.weight,     torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.9.attn_k.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.9.attn_q.bias,         torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.9.attn_v.bias,         torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.22.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.22.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.23.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.23.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.23.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.23.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.23.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.24.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.24.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.24.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.24.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.24.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.25.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.25.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.25.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.25.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.25.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.26.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.26.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.26.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.26.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.26.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.27.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.27.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.27.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.27.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.27.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.28.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.28.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.28.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.28.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.28.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.29.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.29.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.29.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.29.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.29.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.30.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.30.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.30.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.30.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.30.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.31.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.31.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.31.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.31.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.31.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.32.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.32.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.32.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.32.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.32.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.33.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.33.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.33.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.33.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.33.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.34.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.34.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.34.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.34.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.34.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.35.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.35.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.35.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.35.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.35.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.36.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.36.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.36.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.36.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.36.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.37.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.37.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.37.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.37.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.37.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.38.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.38.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.38.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.38.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.38.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.39.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.39.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.39.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.39.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.39.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.40.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.40.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.40.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.40.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.40.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.41.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.41.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.41.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.41.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.41.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.42.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.42.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.42.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.42.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.42.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.43.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.43.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.43.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.43.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.43.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.44.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.44.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.44.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.44.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.44.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.45.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.45.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.45.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.45.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.45.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.46.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.46.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.46.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.46.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.46.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.47.attn_norm.weight,   torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.47.ffn_norm.weight,    torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.47.attn_k.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:blk.47.attn_q.bias,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:blk.47.attn_v.bias,        torch.float16 --> F32, shape = {1024}
INFO:hf-to-gguf:output_norm.weight,        torch.float16 --> F32, shape = {5120}
INFO:hf-to-gguf:output.weight,             torch.float16 --> Q8_0, shape = {5120, 152064}
INFO:hf-to-gguf:blk.0.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.0.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.0.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.0.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.0.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.0.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.0.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.1.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.1.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.1.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.1.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.1.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.1.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.1.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.2.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.2.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.2.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.2.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.2.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.2.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.2.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.3.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.3.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.3.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.3.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.3.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.3.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.3.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.4.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.4.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.4.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.4.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.4.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.4.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.4.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.5.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.5.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.5.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.5.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.5.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.5.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.5.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.6.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.6.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.6.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.6.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.6.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.6.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.6.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.7.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.7.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.7.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.7.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.7.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.7.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.7.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.8.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.8.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.8.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.8.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.8.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.10.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.10.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.10.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.10.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.10.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.10.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.10.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.11.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.11.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.11.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.11.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.11.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.11.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.11.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.12.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.12.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.12.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.12.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.12.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.12.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.12.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.13.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.13.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.13.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.13.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.13.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.13.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.13.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.14.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.14.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.14.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.14.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.14.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.14.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.14.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.15.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.15.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.15.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.15.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.15.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.15.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.15.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.16.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.16.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.16.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.16.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.16.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.16.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.16.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.17.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.17.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.17.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.17.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.17.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.17.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.17.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.18.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.18.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.18.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.18.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.18.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.18.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.18.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.19.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.19.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.19.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.19.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.19.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.19.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.19.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.20.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.20.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.20.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.20.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.20.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.20.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.20.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.21.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.21.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.21.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.21.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.21.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.21.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.21.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.22.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.22.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.22.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.22.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.22.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.8.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.8.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.9.ffn_down.weight,     torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.9.ffn_gate.weight,     torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.9.ffn_up.weight,       torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.9.attn_k.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.9.attn_output.weight,  torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.9.attn_q.weight,       torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.9.attn_v.weight,       torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.22.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.22.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.23.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.23.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.23.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.23.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.23.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.23.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.23.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.24.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.24.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.24.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.24.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.24.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.24.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.24.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.25.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.25.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.25.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.25.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.25.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.25.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.25.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.26.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.26.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.26.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.26.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.26.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.26.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.26.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.27.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.27.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.27.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.27.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.27.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.27.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.27.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.28.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.28.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.28.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.28.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.28.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.28.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.28.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.29.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.29.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.29.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.29.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.29.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.29.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.29.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.30.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.30.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.30.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.30.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.30.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.30.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.30.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.31.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.31.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.31.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.31.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.31.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.31.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.31.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.32.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.32.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.32.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.32.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.32.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.32.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.32.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.33.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.33.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.33.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.33.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.33.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.33.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.33.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.34.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.34.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.34.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.34.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.34.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.34.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.34.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.35.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.35.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.35.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.35.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.35.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.35.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.35.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.36.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.36.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.36.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.36.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.36.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.36.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.36.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.37.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.37.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.37.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.37.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.37.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.37.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.37.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.38.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.38.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.38.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.38.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.38.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.38.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.38.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.39.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.39.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.39.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.39.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.39.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.39.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.39.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.40.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.40.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.40.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.40.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.40.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.40.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.40.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.41.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.41.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.41.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.41.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.41.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.41.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.41.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.42.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.42.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.42.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.42.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.42.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.42.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.42.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.43.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.43.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.43.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.43.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.43.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.43.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.43.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.44.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.44.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.44.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.44.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.44.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.44.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.44.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.45.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.45.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.45.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.45.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.45.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.45.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.45.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.46.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.46.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.46.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.46.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.46.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.46.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.46.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.47.ffn_down.weight,    torch.float32 --> Q8_0, shape = {13824, 5120}
INFO:hf-to-gguf:blk.47.ffn_gate.weight,    torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.47.ffn_up.weight,      torch.float32 --> Q8_0, shape = {5120, 13824}
INFO:hf-to-gguf:blk.47.attn_k.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:blk.47.attn_output.weight, torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.47.attn_q.weight,      torch.float32 --> Q8_0, shape = {5120, 5120}
INFO:hf-to-gguf:blk.47.attn_v.weight,      torch.float32 --> Q8_0, shape = {5120, 1024}
INFO:hf-to-gguf:Set meta model
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:gguf: context length = 32768
INFO:hf-to-gguf:gguf: embedding length = 5120
INFO:hf-to-gguf:gguf: feed forward length = 13824
INFO:hf-to-gguf:gguf: head count = 40
INFO:hf-to-gguf:gguf: key-value head count = 8
WARNING:hf-to-gguf:Unknown RoPE type: default
INFO:hf-to-gguf:gguf: rope scaling type = NONE
INFO:hf-to-gguf:gguf: rope theta = 1000000.0
INFO:hf-to-gguf:gguf: rms norm epsilon = 1e-06
INFO:hf-to-gguf:gguf: file type = 7
INFO:hf-to-gguf:Set model quantization version
INFO:hf-to-gguf:Set model tokenizer
INFO:gguf.vocab:Adding 151387 merge(s).
INFO:gguf.vocab:Setting special token type eos to 151645
INFO:gguf.vocab:Setting special token type pad to 151643
INFO:gguf.vocab:Setting special token type bos to 151643
INFO:gguf.vocab:Setting add_bos_token to False
INFO:gguf.vocab:Setting chat_template to {%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}

INFO:gguf.gguf_writer:Writing the following files:
INFO:gguf.gguf_writer:qwen2.5-14b-instruct-q8_0.gguf: n_tensors = 579, total_size = 15.7G
Writing:   0%|                                                                                                                                                        | 0.00/15.7G [00:00<?, ?byte/s]

Writing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15.7G/15.7G [07:20<00:00, 35.7Mbyte/s]
INFO:hf-to-gguf:Model successfully exported to qwen2.5-14b-instruct-q8_0.gguf
(llama_inf) shi19@AIPC:~/LLM_Exec$
(llama_inf) shi19@AIPC:~/LLM_Exec$
(llama_inf) shi19@AIPC:~/LLM_Exec$ ls -lsh
total 43G
4.0K drwxr-xr-x 3 shi19 shi19 4.0K Nov  4 09:59 Qwen2.5-14B-Instruct-GPTQ-Int8
 28G -rw-r--r-- 1 shi19 shi19  28G Nov  4 10:23 qwen2.5-14b-instruct-f16.gguf
 15G -rw-r--r-- 1 shi19 shi19  15G Nov  4 10:33 qwen2.5-14b-instruct-q8_0.gguf

````

# 测试流程

----



## 简单摸底

----



```bash
#CPU侧执行：

./bin/llama-cli -m ~/Qwen2.5-0.5B-Instruct-Q2_K.gguf -p "hello, my name is jack" -n 128 -no-cnv

#GPU侧执行：
./bin/llama-cli -m ~/Qwen2.5-0.5B-Instruct-Q2_K.gguf -p "hello, my name is jack" -n 128 -no-cnv -ngl 29
```



```bash
(llama_inf) shi19@AIPC:~/LLM_Exec$ llama-cli -m ./qwen2.5-14b-instruct-q8_0.gguf -p "hello, my name is jack" -n 128 -no-cnv -ngl 29
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4080 Laptop GPU, compute capability 8.9, VMM: yes
--no-conversation is not supported by llama-cli
please use llama-completion instead

Loading model...


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8211-a0ed91a44
model      : qwen2.5-14b-instruct-q8_0.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> hello, my name is jack

Hello Jack! It's nice to meet you. How can I assist you today?

[ Prompt: 45.2 t/s | Generation: 8.1 t/s ]

```



### 结果解析

这条命令整体是在用 **llama.cpp 的 `llama-cli`**，以 **CUDA GPU 加速** 的方式，加载本地 Qwen2.5-14B Q8_0 GGUF 模型，对 `"hello, my name is jack"` 做一次生成测试，最多生成 128 个 token，并关闭对话模式。211

#### **逐参数含义**

|                 片段                  |                             含义                             |
| :-----------------------------------: | :----------------------------------------------------------: |
|              `llama-cli`              | llama.cpp 的命令行推理工具，适合交互式测试或单次 prompt 推理2 |
| `-m ./qwen2.5-14b-instruct-q8_0.gguf` |                  指定本地 GGUF 模型文件路径                  |
|     `-p "hello, my name is jack"`     |             输入提示词，模型会基于这句话继续生成             |
|               `-n 128`                |                   最多生成 128 个 token11                    |
|               `-no-cnv`               |                尝试关闭 conversation 对话模式                |
|               `-ngl 29`               |                把模型前 29 层卸载到 GPU 推理2                |

#### **输出逐段解读**

```text
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4080 Laptop GPU, compute capability 8.9, VMM: yes
```

说明 llama.cpp 已经成功识别到一块 **RTX 4080 Laptop GPU**，并且启用了 CUDA 后端。
这表示你的编译版本支持 CUDA，GPU 推理链路是通的

```
--no-conversation is not supported by llama-cli
please use llama-completion instead
```

这是一个**参数警告**。
新版 llama.cpp 中，`--no-conversation` 不再被 `llama-cli` 支持，建议改用 `llama-completion`。2
不过这条警告不会导致程序失败，后面的推理仍然正常执行了

```
Loading model...
```

正在从磁盘加载 GGUF 模型，并根据 `-ngl 29` 把部分层卸载到 GPU。

```text
> hello, my name is jack

Hello Jack! It's nice to meet you. How can I assist you today?
```

模型成功完成了一次推理：

- `>` 后面是你的输入 prompt
- 下面是模型生成的回复

```
[ Prompt: 45.2 t/s | Generation: 8.1 t/s ]
```

这是性能统计：1

- **Prompt: 45.2 t/s**
  表示输入 prompt 处理速度约为 **45.2 tokens/秒**，也就是 prefill 阶段吞吐。
- **Generation: 8.1 t/s**
  表示后续 token 生成速度约为 **8.1 tokens/秒**，也就是 decode 阶段吞吐。

### **结果评判**

对于 **Qwen2.5-14B Q8_0** 来说，`8.1 t/s` 属于**能跑通，但偏慢**的水平。
原因很可能在于 `-ngl 29` 没有把全部层放到 GPU 上。Qwen2.5-14B 通常有 **48 层左右**，`-ngl 29` 只卸载了约一半层，剩余层仍在 CPU 上计算，所以生成速度会被拖慢。

如果你的显存足够，建议改成：

```bash
-ngl 99
```

或者：

bash

```bash
-ngl 999
```

这样会尽量把全部层都卸载到 GPU，生成速度通常会明显提升。

### **建议改写命令**

如果你只是想测试单次生成性能，可以改成:

```bash
llama-cli \
  -m ./qwen2.5-14b-instruct-q8_0.gguf \
  -p "hello, my name is jack" \
  -n 128 \
  -ngl 99
```

如果新版 llama.cpp 仍然提示 `--no-conversation is not supported`，可以忽略 `-no-cnv`；如果确实需要非交互式单次推理，可以考虑改用：

```bash
llama-completion \
  -m ./qwen2.5-14b-instruct-q8_0.gguf \
  -p "hello, my name is jack" \
  -n 128 \
  -ngl 99
```

目前这条命令已经证明：**模型加载正常、CUDA 可用、推理链路正常**，只是 GPU 卸载层数偏低，生成速度还有提升空间。

------



## llama-completion

----

```bash
(llama_inf) shi19@AIPC:~/LLM_Exec$ llama-completion -m ./qwen2.5-14b-instruct-q8_0.gguf -p "hello, my name is jack" -n 128 -no-cnv  -ngl 999

common_perf_print:    sampling time =      51.81 ms
common_perf_print:    samplers time =      15.78 ms /   134 tokens
common_perf_print:        load time =    6461.20 ms
common_perf_print: prompt eval time =     792.32 ms /     6 tokens (  132.05 ms per token,     7.57 tokens per second)
common_perf_print:        eval time =   72175.57 ms /   127 runs   (  568.31 ms per token,     1.76 tokens per second)
common_perf_print:       total time =   73032.40 ms /   133 tokens
common_perf_print: unaccounted time =      12.70 ms /   0.0 %      (total - sampling - prompt eval - eval) / (total)
common_perf_print:    graphs reused =        126
llama_memory_breakdown_print: | memory breakdown [MiB]          | total   free     self   model   context   compute       unaccounted |
llama_memory_breakdown_print: |   - CUDA0 (RTX 4080 Laptop GPU) | 12281 =    0 + (15296 = 14179 +     768 +     349) + 17592186041400 |
llama_memory_breakdown_print: |   - Host                        |                   816 =   788 +       0 +      28                   |
```



### 结果解析

这段输出主要分三块：**采样/推理耗时**、**内存占用**、**计算图复用**。下面按块帮你拆。

#### **1. 时间指标**



|        指标        |           数值           |                     含义                     |
| :----------------: | :----------------------: | :------------------------------------------: |
|    `load time`     |        6461.20 ms        |         模型加载耗时，约 **6.46 秒**         |
| `prompt eval time` |   792.32 ms / 6 tokens   |    处理 prompt 耗时，约 **7.57 tokens/s**    |
|    `eval time`     |  72175.57 ms / 127 runs  | 生成 127 个 token 耗时，约 **1.76 tokens/s** |
|    `total time`    | 73032.40 ms / 133 tokens |            总耗时约 **73.03 秒**             |
|  `sampling time`   |         51.81 ms         |                  采样总耗时                  |
|  `samplers time`   |  15.78 ms / 134 tokens   |          每次采样平均约 **0.12 ms**          |
| `unaccounted time` |         12.70 ms         |  未被分类的额外开销，占比 **0.0%**，可忽略   |
|  `graphs reused`   |           126            |             计算图复用了 126 次              |

**2. 内存占用**

```
CUDA0 (RTX 4080 Laptop GPU): total 12281 MiB
  self 15296 = model 14179 + context 768 + compute 349
Host: 816 MiB = 788 free + 0 self + 28 model
```

解读：

- **模型权重**：约 **14179 MiB ≈ 13.85 GiB**，基本都在 GPU 上。
- **KV cache / context**：约 **768 MiB**。
- **计算临时占用**：约 **349 MiB**。
- **GPU 总占用**：约 **12281 MiB ≈ 12.0 GiB**。
- **Host 内存**：约 **816 MiB**，其中模型相关仅 28 MiB，说明绝大部分权重已成功卸载到 GPU。

这里 `self 15296` 大于 `total 12281`，通常是因为 llama.cpp 对显存统计做了分类汇总，`self` 是各模块加总，`total` 是实际占用量。

**3. 关键结论**

这次推理 **已经成功把模型全部卸载到 GPU**，显存占用正常，模型加载也正常。
但 **生成速度只有 1.76 tokens/s**，比之前 `-ngl 29` 时的 8.1 t/s 还慢，这不太符合预期。

可能原因：

- `-no-cnv` 在新版 `llama-completion` 中可能仍然不被识别，导致参数未生效；
- 当前 prompt 只有 6 个 token，但生成了 127 个 token，单次运行时间波动较大；
- 采样、调度或上下文长度设置可能影响了吞吐。

#### **建议验证命令**

可以再做一次对照测试：

```bash
llama-completion \
  -m ./qwen2.5-14b-instruct-q8_0.gguf \
  -p "hello, my name is jack" \
  -n 128 \
  -ngl 999 \
  -b 512
```

如果速度仍然只有 1~2 t/s，可以再检查：

```
llama-completion --help
```

确认当前版本是否仍支持 `-no-cnv`，以及是否有 `--gpu-layers`、`--batch-size` 等参数。



## 阶段二

----



```bash 
#!/bin/bash
#=============================================================
# llama.cpp 本地推理性能评测脚本
# 适用平台：国产 GPU/NPU（昇腾、寒武纪、摩尔线程等）
# 功能：测试 prefill / decode / 端到端推理性能
#=============================================================

set -e

# ==================== 配置区 ====================
# 模型路径（GGUF 格式）
MODEL_PATH="${MODEL_PATH:-./models/qwen2.5-14b-instruct-q8_0.gguf}"

# llama.cpp 可执行文件路径
LLAMA_CLI="${LLAMA_CLI:-./llama.cpp/build/bin/llama-cli}"
LLAMA_BENCH="${LLAMA_BENCH:-./llama.cpp/build/bin/llama-bench}"

# 评测参数
PROMPT_SIZE="${PROMPT_SIZE:-512}"        # 输入 prompt token 数
MAX_TOKENS="${MAX_TOKENS:-256}"          # 最大生成 token 数
BATCH_SIZE="${BATCH_SIZE:-512}"          # batch size
GPU_LAYERS="${GPU_LAYERS:-999}"          # offload 到 GPU 的层数（999=全部）
THREADS="${THREADS:-8}"                  # CPU 线程数
REPEAT="${REPEAT:-3}"                    # 每组测试重复次数

# 输出文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PLATFORM_NAME=$(uname -m)
OUTPUT_DIR="./benchmark_results/${TIMESTAMP}_${PLATFORM_NAME}"
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

# ==================== 颜色输出 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $1"; }

# ==================== 环境检测 ====================
detect_platform() {
    log_step "检测硬件平台..."
    echo "============================================" >> "$SUMMARY_FILE"
    echo "  推理性能评测报告" >> "$SUMMARY_FILE"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
    echo "============================================" >> "$SUMMARY_FILE"

    # 系统信息
    echo "" >> "$SUMMARY_FILE"
    echo "--- 系统信息 ---" >> "$SUMMARY_FILE"
    echo "OS: $(uname -s) $(uname -r)" >> "$SUMMARY_FILE"
    echo "Arch: $(uname -m)" >> "$SUMMARY_FILE"
    echo "CPU: $(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo 'unknown')" >> "$SUMMARY_FILE"
    echo "CPU Cores: $(nproc 2>/dev/null || echo 'unknown')" >> "$SUMMARY_FILE"

    # 内存信息
    TOTAL_MEM=$(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo "unknown")
    echo "Total RAM: ${TOTAL_MEM}" >> "$SUMMARY_FILE"

    # GPU 检测（适配多种国产 GPU）
    echo "" >> "$SUMMARY_FILE"
    echo "--- GPU/NPU 信息 ---" >> "$SUMMARY_FILE"

    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> "$SUMMARY_FILE" 2>/dev/null || true
    fi

    # 昇腾 NPU
    if command -v npu-smi &>/dev/null; then
        npu-smi info 2>/dev/null | head -20 >> "$SUMMARY_FILE" || true
    fi

    # 寒武纪
    if command -v cnmon &>/dev/null; then
        cnmon 2>/dev/null | head -20 >> "$SUMMARY_FILE" || true
    fi

    # 摩尔线程
    if command -v mthreads-gmi &>/dev/null; then
        mthreads-gmi 2>/dev/null | head -20 >> "$SUMMARY_FILE" || true
    fi

    echo "" >> "$SUMMARY_FILE"
    echo "--- 模型信息 ---" >> "$SUMMARY_FILE"
    echo "Model: ${MODEL_PATH}" >> "$SUMMARY_FILE"
    echo "GPU Layers: ${GPU_LAYERS}" >> "$SUMMARY_FILE"
    echo "Batch Size: ${BATCH_SIZE}" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    log_info "平台检测完成，报告写入 ${SUMMARY_FILE}"
}

# ==================== 前置检查 ====================
preflight_check() {
    log_step "前置检查..."

    if [ ! -f "$MODEL_PATH" ]; then
        log_error "模型文件不存在: ${MODEL_PATH}"
        log_info "请设置 MODEL_PATH 环境变量指向你的 GGUF 模型文件"
        exit 1
    fi

    if [ ! -x "$LLAMA_CLI" ]; then
        log_error "llama-cli 不存在或不可执行: ${LLAMA_CLI}"
        log_info "请编译 llama.cpp 或设置 LLAMA_CLI 环境变量"
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"
    log_info "结果将保存到: ${OUTPUT_DIR}"
}

# ==================== 测试 1: llama-bench 标准评测 ====================
run_llama_bench() {
    if [ ! -x "$LLAMA_BENCH" ]; then
        log_warn "llama-bench 不可用，跳过标准评测"
        return
    fi

    log_step "运行 llama-bench 标准评测..."
    echo "--- llama-bench 标准评测 ---" >> "$SUMMARY_FILE"

    local bench_output="${OUTPUT_DIR}/llama_bench_raw.txt"

    # 测试不同 prompt 长度下的 prefill 速度
    "$LLAMA_BENCH" \
        -m "$MODEL_PATH" \
        -p 128,256,512,1024,2048 \
        -n 128 \
        -b "$BATCH_SIZE" \
        -t "$THREADS" \
        -ngl "$GPU_LAYERS" \
        -r "$REPEAT" \
        2>&1 | tee "$bench_output"

    echo "" >> "$SUMMARY_FILE"
    cat "$bench_output" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    log_info "llama-bench 评测完成"
}

# ==================== 测试 2: 端到端推理评测 ====================
run_e2e_test() {
    log_step "运行端到端推理评测..."
    echo "--- 端到端推理评测 ---" >> "$SUMMARY_FILE"

    # 生成指定长度的 prompt
    local prompt_file="${OUTPUT_DIR}/prompt.txt"
    python3 -c "
import random, string
# 生成约 PROMPT_SIZE 个 token 的文本（粗略估计 1 token ≈ 4 字符）
target_chars = ${PROMPT_SIZE} * 4
words = []
total = 0
while total < target_chars:
    w = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3,8)))
    words.append(w)
    total += len(w) + 1
print(' '.join(words))
" > "$prompt_file"

    local prompt_text
    prompt_text=$(cat "$prompt_file")

    # 多轮测试
    for i in $(seq 1 "$REPEAT"); do
        log_info "端到端测试 第 ${i}/${REPEAT} 轮"
        local result_file="${OUTPUT_DIR}/e2e_run_${i}.txt"

        local start_time
        start_time=$(date +%s%N)

        "$LLAMA_CLI" \
            -m "$MODEL_PATH" \
            -p "$prompt_text" \
            -n "$MAX_TOKENS" \
            -b "$BATCH_SIZE" \
            -t "$THREADS" \
            -ngl "$GPU_LAYERS" \
            --color \
            -no-cnv \
            2>&1 | tee "$result_file"

        local end_time
        end_time=$(date +%s%N)
        local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

        echo "Run ${i}: total_time=${elapsed_ms}ms" >> "$SUMMARY_FILE"

        # 提取性能指标
        local pp_speed=$(grep -oP 'pp\s*.*?\K[\d.]+' "$result_file" 2>/dev/null || echo "N/A")
        local tg_speed=$(grep -oP 'tg\s*.*?\K[\d.]+' "$result_file" 2>/dev/null || echo "N/A")

        echo "  Prompt Processing: ${pp_speed} tokens/s" >> "$SUMMARY_FILE"
        echo "  Token Generation:  ${tg_speed} tokens/s" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    done

    log_info "端到端推理评测完成"
}

# ==================== 测试 3: 不同并发长度评测 ====================
run_context_length_test() {
    log_step "运行不同上下文长度评测..."
    echo "--- 不同上下文长度评测 ---" >> "$SUMMARY_FILE"
    printf "%-15s %-15s %-15s %-15s\n" "Context Len" "PP (t/s)" "TG (t/s)" "Total (s)" >> "$SUMMARY_FILE"
    printf "%-15s %-15s %-15s %-15s\n" "-----------" "--------" "--------" "---------" >> "$SUMMARY_FILE"

    for ctx_len in 512 1024 2048 4096; do
        log_info "测试上下文长度: ${ctx_len}"
        local result_file="${OUTPUT_DIR}/ctx_${ctx_len}.txt"

        local start_time
        start_time=$(date +%s%N)

        "$LLAMA_CLI" \
            -m "$MODEL_PATH" \
            -p "$(python3 -c "print('hello ' * ${ctx_len})")" \
            -n 64 \
            -b "$BATCH_SIZE" \
            -t "$THREADS" \
            -ngl "$GPU_LAYERS" \
            -c "$ctx_len" \
            -no-cnv \
            2>&1 | tee "$result_file"

        local end_time
        end_time=$(date +%s%N)
        local elapsed_s
        elapsed_s=$(echo "scale=2; ($end_time - $start_time) / 1000000000" | bc)

        local pp_speed=$(grep -oP 'pp\s*.*?\K[\d.]+' "$result_file" 2>/dev/null || echo "N/A")
        local tg_speed=$(grep -oP 'tg\s*.*?\K[\d.]+' "$result_file" 2>/dev/null || echo "N/A")

        printf "%-15s %-15s %-15s %-15s\n" "$ctx_len" "$pp_speed" "$tg_speed" "$elapsed_s" >> "$SUMMARY_FILE"
    done

    echo "" >> "$SUMMARY_FILE"
    log_info "上下文长度评测完成"
}

# ==================== 测试 4: 显存/内存占用监控 ====================
run_memory_test() {
    log_step "测试模型加载内存占用..."
    echo "--- 内存占用 ---" >> "$SUMMARY_FILE"

    # 加载模型并观察内存
    local mem_log="${OUTPUT_DIR}/memory_usage.txt"

    # 后台监控内存
    (
        while true; do
            echo "$(date +%s) $(free -m | awk '/Mem:/{print $3}')" >> "$mem_log"
            sleep 1
        done
    ) &
    local mem_pid=$!

    # 运行一次短推理
    "$LLAMA_CLI" \
        -m "$MODEL_PATH" \
        -p "Hello, please introduce yourself briefly." \
        -n 32 \
        -t "$THREADS" \
        -ngl "$GPU_LAYERS" \
        -no-cnv \
        > /dev/null 2>&1

    # 停止监控
    kill "$mem_pid" 2>/dev/null || true
    wait "$mem_pid" 2>/dev/null || true

    # 统计峰值内存
    if [ -f "$mem_log" ]; then
        local peak_mem
        peak_mem=$(awk '{print $2}' "$mem_log" | sort -n | tail -1)
        echo "Peak RAM Usage: ${peak_mem} MB" >> "$SUMMARY_FILE"
    fi
    echo "" >> "$SUMMARY_FILE"

    log_info "内存占用测试完成"
}

# ==================== 生成汇总报告 ====================
generate_report() {
    log_step "生成评测汇总报告..."

    echo "" >> "$SUMMARY_FILE"
    echo "============================================" >> "$SUMMARY_FILE"
    echo "  评测完成: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
    echo "============================================" >> "$SUMMARY_FILE"

    echo ""
    log_info "========================================="
    log_info "  评测完成！"
    log_info "  结果目录: ${OUTPUT_DIR}"
    log_info "  汇总报告: ${SUMMARY_FILE}"
    log_info "========================================="
    echo ""

    # 打印汇总
    cat "$SUMMARY_FILE"
}

# ==================== 主流程 ====================
main() {
    echo ""
    log_info "╔══════════════════════════════════════╗"
    log_info "║   llama.cpp 推理性能评测工具        ║"
    log_info "║   适配国产 GPU/NPU 平台             ║"
    log_info "╚══════════════════════════════════════╝"
    echo ""

    preflight_check
    detect_platform

    # 按顺序执行各项测试
    run_llama_bench
    run_e2e_test
    run_context_length_test
    run_memory_test

    generate_report
}

# 支持命令行参数覆盖
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)     MODEL_PATH="$2"; shift 2 ;;
        -n|--tokens)    MAX_TOKENS="$2"; shift 2 ;;
        -p|--prompt)    PROMPT_SIZE="$2"; shift 2 ;;
        -b|--batch)     BATCH_SIZE="$2"; shift 2 ;;
        -t|--threads)   THREADS="$2"; shift 2 ;;
        -ngl|--gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        -r|--repeat)    REPEAT="$2"; shift 2 ;;
        --cli-path)     LLAMA_CLI="$2"; shift 2 ;;
        --bench-path)   LLAMA_BENCH="$2"; shift 2 ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -m, --model PATH        GGUF 模型文件路径"
            echo "  -n, --tokens NUM        最大生成 token 数 (默认: 256)"
            echo "  -p, --prompt NUM        输入 prompt token 数 (默认: 512)"
            echo "  -b, --batch NUM         Batch size (默认: 512)"
            echo "  -t, --threads NUM       CPU 线程数 (默认: 8)"
            echo "  -ngl, --gpu-layers NUM  GPU offload 层数 (默认: 999)"
            echo "  -r, --repeat NUM        每组测试重复次数 (默认: 3)"
            echo "  --cli-path PATH         llama-cli 路径"
            echo "  --bench-path PATH       llama-bench 路径"
            echo "  -h, --help              显示帮助"
            exit 0
            ;;
        *) log_error "未知参数: $1"; exit 1 ;;
    esac
done

main
```





## 附录A-参考教程

`Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8` 是 **HuggingFace / GPTQ 量化权重目录**，不是 GGUF 文件，不能直接给 llama.cpp 用。9
需要先用 llama.cpp 的 `convert_hf_to_gguf.py` 转成 GGUF，再根据需要选择 `q8_0`。

### **推荐流程：HF/GPTQ 目录 → F16 GGUF → Q8_0 GGUF**

```bash
# 1. 准备 llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 2. 安装转换依赖
pip install -r requirements/requirements-convert_hf_to_gguf.txt

# 3. 编译 C++ 工具，生成 llama-quantize
cmake -B build
cmake --build build --config Release
```

然后执行转换:

```bash
# 4. 先转成 F16 GGUF（中间文件）
python convert_hf_to_gguf.py \
    ../Qwen2.5-14B-Instruct-GPTQ-Int8 \
    --outtype f16 \
    --outfile qwen2.5-14b-instruct-f16.gguf
```

再量化成 `q8_0`：

```bash
# 5. 量化成 Q8_0 GGUF
./build/bin/llama-quantize \
    qwen2.5-14b-instruct-f16.gguf \
    qwen2.5-14b-instruct-q8_0.gguf \
    Q8_0
```

最后把脚本里的模型路径改成：

```bash
MODEL_PATH="./qwen2.5-14b-instruct-q8_0.gguf"
```

### **也可以一步转成 Q8_0**

部分 llama.cpp 版本支持：

```bash
python convert_hf_to_gguf.py \
    ../Qwen2.5-14B-Instruct-GPTQ-Int8 \
    --outtype q8_0 \
    --outfile qwen2.5-14b-instruct-q8_0.gguf
```

但如果报 `--outtype q8_0 not supported`，就用前面的“先 f16，再 llama-quantize”两步法。11

### **注意两点**

1. **GPTQ-Int8 不是 GGUF-Q8_0**
   你下载的模型已经是 GPTQ Int8 权重，但 llama.cpp 的 GGUF 量化体系是另一套。所以更稳妥的做法是：把它当作 HF 模型目录，转成 GGUF 后重新用 llama.cpp 的 `Q8_0` 量化。9
2. **14B 模型转换比较吃内存**
   建议保证有足够内存和磁盘空间；如果转换时 OOM，可以先加 swap，或换一台内存更大的机器执行转换
