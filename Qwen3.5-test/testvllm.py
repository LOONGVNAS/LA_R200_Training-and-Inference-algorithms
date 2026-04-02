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
