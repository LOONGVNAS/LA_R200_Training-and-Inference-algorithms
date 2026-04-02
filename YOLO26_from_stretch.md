# YOLO26 验证流程 v2 — 针对你的实际环境

> **精确对齐的栈**
>
> ```
> OS          Ubuntu 24.04 (WSL2)
> conda       25.11.1
> Python      3.13.11          ← base 环境已有，无需另建
> Driver      581.83 (Win侧)   → CUDA 上限 13.0
> GPU         RTX 4080 12 GB
> PyTorch     2.10.0           ← 最新 stable，原生支持 Python 3.13 + cu130
> Ultralytics ≥ 8.4.0          ← YOLO26 最低版本要求
> ```

---

## ⚡ 核心决策：不用新建环境

上一版建议你用 Python 3.10 新建环境。**现在不需要了。**

PyTorch 2.10（2026-01-21 发布）已经：

- 原生支持 Python 3.13（含 `torch.compile`）
- 提供了 **CUDA 13.0 轮包**（`cu130`），恰好匹配你 `nvidia-smi` 显示的驱动上限

你的 `base` 环境直接用就行。唯一建议是为 YOLO26 开一个干净的 conda 环境，避免和其他项目冲突，但 Python 版本不需要降。

---

## 第一步：创建环境（保持 Python 3.13）

```bash
# 用你现有的 Python 3.13 创建一个隔离环境
conda create -n yolo26 python=3.13 -y
conda activate yolo26
```

如果你不介意装到 base 里，可以跳过这一步，直接往下走。

---

## 第二步：装 PyTorch 2.10 + CUDA 13.0

这是整个流程里**最关键的一行命令**。版本和 index-url 都要精确。

```bash
pip install torch==2.10.0 torchvision==0.21.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130
```

**版本对应关系（不能乱配）：**

| torch  | torchvision | torchaudio | CUDA   |
| ------ | ----------- | ---------- | ------ |
| 2.10.0 | 0.21.0      | 2.10.0     | 13.0 ✅ |

**装完，立刻验证——这一步通了后面才能往下走：**

```bash
python -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:          {torch.cuda.get_device_name(0)}')
    print(f'CUDA ver:     {torch.version.cuda}')
    print(f'显存:         {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    # 跑一个小计算确认 GPU 真的能用
    a = torch.rand(1000, 1000, device='cuda')
    b = torch.rand(1000, 1000, device='cuda')
    c = a @ b
    print(f'矩阵乘法测试: PASSED ✅')
"
```

期望输出：

```
PyTorch:        2.10.0+cu130
CUDA available: True
GPU:            NVIDIA GeForce RTX 4080 ...
CUDA ver:       13.0
显存:           12.2 GB
矩阵乘法测试:   PASSED ✅
```

> 🛑 如果 `CUDA available: False`，**停在这里**，后面装什么都没用。最常见的原因是装了 CPU 版的 torch（url 写错了或者没写）。用 `pip show torch` 确认版本尾部有 `+cu130`。

---

## 第三步：装 Ultralytics（YOLO26 要求 ≥ 8.4.0）

```bash
pip install ultralytics>=8.4.0
```

装完确认版本：

```bash
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
# 期望：8.4.x 或更高
```

---

## 第四步：第一次推断 — 确认模型能跑

```bash
# 最简单的一行，能跑说明整条链路通
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg' device=0 save=True
```

如果网络访问图片有问题，换本地图片路径。结果默认存到 `runs/detect/predict/`。

用 Python 也行，更方便看详细输出：

```python
# predict_test.py
from ultralytics import YOLO

model = YOLO("yolo26n.pt")   # 首次会自动下载权重 (~3 MB)
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=0,
    imgsz=640,
    conf=0.25,
    save=True
)

for r in results:
    print(f"\n检测到 {len(r.boxes)} 个目标:")
    for box in r.boxes:
        cls  = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"  {cls:20s} conf={conf:.3f}")
```

---

## 第五步：性能基测 — 你的 4080 能跑多快

```python
# benchmark.py
import time
import torch
from ultralytics import YOLO

MODELS   = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
IMGSZ    = 640
WARMUP   = 10
RUNS     = 100

print(f"{'模型':<14} {'推断(ms)':<12} {'FPS':<10} {'显存峰值(MB)'}")
print("-" * 52)

for name in MODELS:
    model = YOLO(name)

    # 用随机图模拟推断（只测速度，不需要真图片）
    dummy = torch.rand(1, 3, IMGSZ, IMGSZ).cuda()

    # Warmup — CUDA 第一次用会有编译延迟
    for _ in range(WARMUP):
        model.predict(source=dummy, device=0, verbose=False)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for _ in range(RUNS):
        model.predict(source=dummy, device=0, verbose=False)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms   = (t1 - t0) / RUNS * 1000
    fps      = 1000.0 / avg_ms
    peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB

    print(f"{name:<14} {avg_ms:<12.2f} {fps:<10.1f} {peak_mem:.0f}")
```

**RTX 4080 上的参考范围（PyTorch backend，非 TensorRT）：**

| 模型    | 预期推断延迟 | 预期 FPS  |
| ------- | ------------ | --------- |
| yolo26n | 2 ~ 4 ms     | 250 ~ 500 |
| yolo26s | 3 ~ 6 ms     | 160 ~ 330 |
| yolo26m | 5 ~ 9 ms     | 110 ~ 200 |
| yolo26l | 8 ~ 14 ms    | 70 ~ 125  |
| yolo26x | 12 ~ 22 ms   | 45 ~ 83   |

> Ultralytics 官方速度表用的是 TensorRT 后端。你上面的数字用的是纯 PyTorch，会慢 1.5～2x，这是正常的。如果要对齐官方数字，导出 TensorRT 格式即可（见第七步）。

---

## 第六步：COCO val 精度验证

这是和论文对齐 mAP 数字的标准手段。会自动下载 COCO val2017（~780 MB）。

```bash
# CLI 版本，最简洁
yolo val model=yolo26n.pt data=coco.yaml device=0 batch=32 split=val
```

或者 Python：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
metrics = model.val(data="coco.yaml", device=0, batch=32, split="val", imgsz=640)

print(f"\nmAP50:     {metrics.box.map50:.1f}")
print(f"mAP50-95:  {metrics.box.map:.1f}")
```

**YOLO26 各规格的 COCO 参考精度：**

| 模型    | mAP50-95 | mAP50 | 参数量 |
| ------- | -------- | ----- | ------ |
| yolo26n | ~40.9    | ~56+  | 2.5M   |
| yolo26s | ~48+     | ~65+  | 9.4M   |
| yolo26m | ~52+     | ~69+  | 25.9M  |
| yolo26l | ~54+     | ~71+  | 49.0M  |
| yolo26x | ~55+     | ~72+  | 68.2M  |

你测出的数字和表格偏差在 ±2 个点之内是正常的（跟图片采样和硬件浮点精度有关）。超过 3 个点就值得排查。

---

## 第七步：导出 — 生产环境格式

YOLO26 去掉了 DFL 模块，导出兼容性比之前大幅提升。

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# ONNX — 跨平台标准，最通用
model.export(format="onnx", imgsz=640)                          # → yolo26n.onnx

# TensorRT — NVIDIA 生产级，速度最快
model.export(format="tensorrt", imgsz=640)                      # → yolo26n.engine

# TensorRT FP16 — 速度翻倍，精度基本不掉
model.export(format="tensorrt", imgsz=640, half=True)

# TensorRT INT8 — 极致性能，需要校准数据
model.export(format="tensorrt", imgsz=640, int8=True, data="coco.yaml")
```

导出后用对应引擎跑推断，确认输出一致：

```python
# 用导出的 ONNX 模型跑
onnx_model = YOLO("yolo26n.onnx")
results = onnx_model.predict("https://ultralytics.com/images/bus.jpg", device=0)
```

---

## 第八步：多任务验证（按需选做）

权重后缀决定了任务类型，不要混用：

```python
from ultralytics import YOLO

# 实例分割（像素级轮廓）
YOLO("yolo26n-seg.pt").predict("image.jpg", device=0, save=True)

# 姿态估计（人体关键点）
YOLO("yolo26n-pose.pt").predict("image.jpg", device=0, save=True)

# 有向检测（航拍/卫星图的旋转目标）
YOLO("yolo26n-obb.pt").predict("image.jpg", device=0, save=True)

# 图像分类
YOLO("yolo26n-cls.pt").predict("image.jpg", device=0, save=True)
```

---

## 常见问题速查

**Q：装好了但 `CUDA available: False`**
A：`pip show torch` 看版本末尾是否有 `+cu130`。没有就是装了 CPU 版，重新用带 `--index-url` 的命令装。

**Q：`nvidia-smi` 看不到 GPU**
A：WSL2 不需要在 Linux 里装驱动。驱动只装 Windows 侧。你的 Driver 581.83 已经支持 CUDA 13.0，不需要动。

**Q：下载权重太慢**
A：权重托管在 GitHub Releases。网慢的话先用浏览器下载 `.pt` 文件，放到当前目录即可，ultralytics 会优先读本地文件。

**Q：OOM（显存不够）**
A：RTX 4080 跑 yolo26x 不会 OOM。如果是训练时 OOM，先降 `batch`，再试 `half=True`（FP16 训练）。

**Q：Python 3.13 还是有某个包不兼容**
A：ultralytics 核心依赖栈（torch 2.10 / numpy / onnx）已经全支持 3.13。如果遇到某个可选依赖不兼容，报错信息会明确指出哪个包，到时候单独处理。

---

## 一键执行版（复制粘贴）

```bash
# ① 创建环境（保持 3.13，可跳过）
conda create -n yolo26 python=3.13 -y && conda activate yolo26

# ② 装 PyTorch 2.10 + CUDA 13.0
pip install torch==2.10.0 torchvision==0.21.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ③ 验证 CUDA
python -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'CUDA FAIL')"

# ④ 装 Ultralytics
pip install ultralytics>=8.4.0

# ⑤ 一行推断
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg' device=0 save=True
```

---

*更新日期: 2026-02-03 | 基于 PyTorch 2.10.0 + Ultralytics 8.4.x*

