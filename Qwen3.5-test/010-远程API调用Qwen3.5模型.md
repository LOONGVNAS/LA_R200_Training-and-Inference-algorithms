## 背景知识
---

通过龙芯3C6000/S 工作站双卡 远程API调用太初的Qwen3.5模型。

### 基础环境信息

OS版本： https://pkg.loongnix.cn/loongnix-server/23.1/isos

### 模型背景信息

vllm远程后端推理，难点是模型的调教。

```bash
export SDAA_VISIBLE_DEVICES="24,25,26,27,28,29,30,31"

vllm serve /data01/models/Qwen3.5-27B
--port 36666 \
--tensor_parallel_size 8 \
--max-model-len 262144 \
--reasoning-parser qwen3 
--enable-auto-tool-choice \
--api-key sk-ab674bac964e4e5fbdd130c67949d845b0637c2a089586ce \
--tool-call-parser qwen3_coder \
--served-model-name Qwen3.5-27B \
--max-num-seqs 32 \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

**原始图片内容**

<img width="684" height="292" alt="0846268bf9c927d67865f1be0ae25a3c" src="https://github.com/user-attachments/assets/1ad49100-5367-4636-9789-a658e02b07c8" />

### curl 命令行请求方式

```bash
curl http://112.53.120.78:36666/v1/chat/completions -H "Authorization: Bearer sk-ab674bac964e4e5fbdd130c67949d845b0637c2a089586ce" -H "Content-Type: application/json" -d '{"model": "Qwen3.5-27B", "messages": [{"role": "user", "content": "你好，你是谁？"}]}'
```

格式友好的展示：
```bash
curl http://112.53.120.78:36666/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxx" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```
### 首字延迟和Token速率测试
---

**v1版本**

```python
import requests
import json
import time

# ================= 配置区域 =================
SERVER_IP = "112.53.120.78"
PORT = "36666"
BASE_URL = f"http://{SERVER_IP}:{PORT}"

# 鉴权配置 (对应 curl 中的 sk-xxx)
API_KEY = "sk-ab674bac964e4e5fbdd130c67949d845b0637c2a089586ce" 

# 模型配置 (对应 curl 中的 model)
MODEL_NAME = "Qwen3.5-27B"

# 测试提示词
PROMPT_TEXT = "你好，你是谁？请简单介绍一下你自己。"
# =============================================

def test_vllm_performance():
    # 1. 构造 API 端点 (使用 chat/completions 接口)
    url = f"{BASE_URL}/v1/chat/completions"
    
    # 2. 构造请求头 (添加 Authorization)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 3. 构造请求体 (使用 messages 格式)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": PROMPT_TEXT}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": True  # 开启流式以测试首字延迟
    }

    print(f"正在连接服务器: {BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    print(f"提示词: {PROMPT_TEXT}")
    print("-" * 30)

    try:
        start_time = time.time()
        first_token_time = None
        response_text_length = 0
        
        # 发送 POST 请求
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            # 检查 HTTP 状态码
            if response.status_code != 200:
                print(f"\n[错误] 请求失败，状态码: {response.status_code}")
                print(f"返回信息: {response.text}")
                return

            print("模型回复:", end=" ", flush=True)
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = line_str[6:] # 去除 'data: ' 前缀
                        
                        if data.strip() == '[DONE]':
                            break
                            
                        try:
                            json_data = json.loads(data)
                            choices = json_data.get('choices', [])
                            if choices:
                                # Chat 接口获取 content 的路径是 delta -> content
                                delta = choices[0].get('delta', {})
                                text = delta.get('content', '')
                                
                                if text:
                                    # --- 计算首字延迟 ---
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                        ttft = (first_token_time - start_time) * 1000
                                        print(f"\n[指标] 首字延迟 (TTFT): {ttft:.2f} ms")
                                    
                                    # 打印文本
                                    print(text, end="", flush=True)
                                    response_text_length += len(text)
                                    
                        except json.JSONDecodeError:
                            continue
            
            # --- 计算总耗时 ---
            end_time = time.time()
            total_duration = end_time - start_time
            
            print("\n" + "-" * 30)
            print(f"[指标] 总生成耗时: {total_duration:.2f} 秒")
            # 注意：流式模式下精确 Token 数较难获取，这里仅做字符级估算
            # 实际 Token 速率 = 总 Token 数 / 总耗时
            
    except requests.exceptions.ConnectionError:
        print(f"\n[错误] 无法连接到服务器 {BASE_URL}。请检查 IP、端口或网络代理。")
    except Exception as e:
        print(f"\n[错误] 发生异常: {e}")

if __name__ == "__main__":
    test_vllm_performance()
```

**v2版本**

```python
import requests
import json
import time

# ================= 配置区域 =================
SERVER_IP = "112.53.120.78"
PORT = "36666"
BASE_URL = f"http://{SERVER_IP}:{PORT}"

# 鉴权配置
API_KEY = "sk-ab674bac964e4e5fbdd130c67949d845b0637c2a089586ce" 
# 模型配置
MODEL_NAME = "Qwen3.5-27B"

# 测试提示词
PROMPT_TEXT = "你好，请介绍一下量子力学的基本原理。"
# =============================================


class VLLMTester:
    def __init__(self, base_url, api_key, model):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.model = model
        self.url = f"{self.base_url}/v1/chat/completions"

    def test_stream_performance(self):
        """
        测试流式模式：重点测首字延迟，同时显示输出
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT_TEXT}],
            "stream": True,
            "temperature": 0.7
        }

        print(f"正在连接服务器：{self.url}")
        print("模型回复：", end="", flush=True)

        try:
            start_time = time.time()
            first_token_time = None
            full_response = ""

            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=60
            )

            if response.status_code != 200:
                print(f"\n[错误] HTTP状态码: {response.status_code}")
                print(response.text)
                return

            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')

                    if line_str.strip() == "data: [DONE]":
                        break

                    if line_str.startswith("data:"):
                        json_str = line_str[5:].strip()
                        try:
                            chunk = json.loads(json_str)
                            delta = chunk['choices'][0]['delta']

                            # --- 核心修改逻辑：同时支持 reason 和 content ---

                            # 1. 优先处理 reasoning_content (思维链)
                            if 'reasoning_content' in delta:
                                content = delta['reasoning_content']
                                # 如果你想区分思考过程，可以加个颜色或前缀，这里直接打印
                                print(content, end="", flush=True)
                                full_response += content

                                # 记录首字延迟（针对推理模型，首字通常指思考的开始）
                                if first_token_time is None:
                                    first_token_time = time.time() - start_time

                            # 2. 处理普通 content
                            elif 'content' in delta:
                                content = delta['content']
                                print(content, end="", flush=True)
                                full_response += content

                                if first_token_time is None:
                                    first_token_time = time.time() - start_time

                        except json.JSONDecodeError:
                            pass
                        except KeyError:
                            pass

            end_time = time.time()
            total_time = end_time - start_time

            print("\n" + "-" * 40)
            if first_token_time:
                print(f"[统计] 首字延迟: {first_token_time:.2f} 秒")
            else:
                print(f"[统计] 未检测到有效数据流")
            print(f"[统计] 总耗时: {total_time:.2f} 秒")
            print(f"[统计] 回复总长度: {len(full_response)} 字符")

        except Exception as e:
            print(f"\n[错误] 请求失败: {str(e)}")

    def test_non_stream_performance(self):
        """
        测试非流式模式：测整体生成速率
        """
        print("\n--- [测试 2/2] 非流式模式 (测生成速率) ---")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT_TEXT}],
            "stream": False,
            "temperature": 0.7
        }

        print("正在生成内容并统计 Token...")
        start_time = time.time()

        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=120
            )

            end_time = time.time()
            total_time = end_time - start_time

            if response.status_code == 200:
                data = response.json()
                # 尝试获取 Token 统计信息
                usage = data.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 'N/A')
                completion_tokens = usage.get('completion_tokens', 'N/A')
                total_tokens = usage.get('total_tokens', 'N/A')

                print(f"[完成] 耗时: {total_time:.2f} 秒")
                print("-" * 40)
                print(f"[统计] 输入 Token 数: {prompt_tokens}")
                print(f"[统计] 输出 Token 数: {completion_tokens}")
                print(f"[统计] 总 Token 数: {total_tokens}")

                if isinstance(completion_tokens, int) and completion_tokens > 0:
                    speed = completion_tokens / total_time
                    print(f"[统计] 生成速率: {speed:.2f} Token/秒")

            else:
                print(f"[错误] 请求失败: {response.status_code}")
                print(response.text)

        except Exception as e:
            print(f"[错误] 请求异常: {str(e)}")


if __name__ == "__main__":
    # 初始化测试器
    tester = VLLMTester(BASE_URL, API_KEY, MODEL_NAME)

    print("--- [测试 1/2] 流式模式 (测首字延迟) ---")
    # 运行流式测试
    tester.test_stream_performance()

    print("\n")

    # 运行非流式测试
    tester.test_non_stream_performance()
```

