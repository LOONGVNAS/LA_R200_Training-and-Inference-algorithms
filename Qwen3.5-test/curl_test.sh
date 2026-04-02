curl http://112.53.120.78:36666/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxx" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'