CORS 是 **Cross-Origin Resource Sharing** 的缩写，中文译为**跨域资源共享**。

## 核心含义

CORS 是一种**浏览器安全机制**，用于控制**不同源（origin）**之间的资源访问权限。

## 为什么需要 CORS

浏览器的**同源策略（Same-Origin Policy）**默认禁止网页向**不同协议、域名或端口**的服务器发起请求，以防止恶意网站窃取数据。

例如：

- `https://a.com` 的网页默认无法直接请求 `https://b.com/api`

CORS 通过在服务器端配置特定的 HTTP 响应头，**有选择地放宽**这一限制，允许合法的跨域访问。

## 关键机制

| 要素                      | 说明                                                         |
| :------------------------ | :----------------------------------------------------------- |
| **预检请求（Preflight）** | 对复杂请求（如 PUT、DELETE、自定义头），浏览器先发送 OPTIONS 探测 |
| **响应头控制**            | `Access-Control-Allow-Origin` 指定允许访问的源               |
| **凭证传递**              | `Access-Control-Allow-Credentials` 控制是否携带 Cookie 等身份凭证 |

## 模型服务中的典型配置

```yaml
# 示例：模型推理 API 的 CORS 配置
enable_cors: true
allow_origins: ["https://your-frontend.com"]  # 白名单，* 表示允许所有（生产环境慎用）
allow_methods: ["POST", "GET", "OPTIONS"]
allow_headers: ["Content-Type", "Authorization"]
allow_credentials: true  # 若需传递用户身份令牌
```

## 安全风险警示

| 配置                                           | 风险                                           |
| :--------------------------------------------- | :--------------------------------------------- |
| `allow_origins: *` + `allow_credentials: true` | **高危**：任何网站均可冒用用户身份调用模型 API |
| 未限制 HTTP 方法                               | 可能暴露 DELETE/PUT 等危险操作                 |
| 未校验 Origin 头                               | 无法抵御 CSRF 类攻击                           |

## 最佳实践

```plain
生产环境务必：
  1. 明确枚举允许的源域名，禁用通配符
  2. 按需开放 HTTP 方法，模型推理通常仅需 POST
  3. 限制请求头白名单，剔除不必要的自定义头
  4. 若启用凭证，必须配对具体的域名白名单
```

简言之，enable CORS 即**授权特定外部网页能够直接调用你的模型服务接口**，是连接前端应用与后端模型推理的桥梁，但配置失当会成为安全敞口。
