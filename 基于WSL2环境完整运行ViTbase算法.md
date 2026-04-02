## 背景信息
---

WSL2+Ubuntu24.04 
```bash
(vit_env) shi19@AIPC:~/ViTBase$ cat /etc/os-release
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo
```

显卡以及cuda信息

```bash
(vit_env) shi19@AIPC:~/ViTBase$ nvidia-smi
Thu Apr  2 19:14:16 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.288.01             Driver Version: 581.83       CUDA Version: 13.0     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4080 ...    On  | 00000000:01:00.0  On |                  N/A |
| N/A   37C    P8               6W / 140W |   1234MiB / 12282MiB |      2%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

## 流程简介
---

环境配置->ViT架构简介->训练代码->推理部署->性能测试->调优建议

## 环境配置
---

**环境信息**

| GPU          | RTX 4080 Laptop  12 GB VRAM |
| ------------ | --------------------------- |
| CUDA Version | 13.0 (Driver 581.83)  WSL2  |
| OS           | Ubuntu 24.04 + WSL2         |
| Conda        | 25.11.1  已安装             |
| 可用 VRAM    | ~11.3 GB (当前空闲)  充足   |

> [!NOTE]
>
>注意
>
>CUDA 13.0 为最新驱动，但 PyTorch 当前支持 CUDA ≤ 12.x。使用 **cu124** 版本的 PyTorch 即可正常运行。

**创建 Conda 环境**

```bash
# 创建独立环境，避免污染现有环境
conda create -n vit_env python=3.11 -y
conda activate vit_env
```

**安装 PyTorch (CUDA 12.4)**

```bash
# cu124 兼容 CUDA 13.x 驱动
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

**安装其他依赖**

```bash
pip install timm transformers datasets \
    accelerate einops matplotlib tqdm \
    tensorboard scikit-learn pillow
```

**验证 GPU 可用**

```python
import torch
print(torch.cuda.is_available())          # True
print(torch.cuda.get_device_name(0))      # RTX 4080 Laptop
print(torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
```

**项目结构**

vit_project/ 

​	├── data/ # 数据集目录 

​	├── checkpoints/ # 模型权重保存 

​	├── logs/ # TensorBoard 日志 

​	├── model.py # ViT 模型定义 

​	├── train.py # 训练脚本 

​	├── dataset.py # 数据加载 

​	└── infer.py # 推理脚本

① 环境配置 — conda 环境创建、PyTorch cu124 安装（兼容你的 CUDA 13.0 驱动）、验证脚本、项目结构

② ViT架构 — 可视化架构图，标注了 Patch Embed → [CLS] token → 12层 Encoder → 分类头的完整流程，以及 ViT-B/16 的关键参数（86M 参数，12GB VRAM 完全够用）

③ 训练代码 — 完整的 model.py、dataset.py（CIFAR-10 示例）、train.py（含 AMP 混合精度、梯度裁剪、TensorBoard、模型保存）

④ 推理部署 — 单张推理、ONNX 导出、HuggingFace 直接使用预训练权重三种方式

⑤ 调优建议 — VRAM 使用估算（推荐 batch=64~128）、梯度累积技巧、WSL2 .wslconfig 配置、常见问题排查

所有代码块支持一键复制。建议从迁移学习（pretrained=True）入手，在 CIFAR-10 上先跑通流程，预期 30 epoch 可达 ~98% 验证精度。

## ViT架构
---

ViT-Base 将图像切分为 16×16 的 patch，展平后加入位置编码，送入标准 Transformer Encoder。

<img width="707" height="334" alt="image" src="https://github.com/user-attachments/assets/83b88457-9b93-4f6e-83d6-0494e81f67c5" />

**ViT-B/16 关键参数**

| 输入分辨率        | 224 × 224               |
| ----------------- | ----------------------- |
| Patch 大小        | 16 × 16 → 196 个 patch  |
| Hidden dim        | 768                     |
| Encoder 层数      | 12 层                   |
| Attention heads   | 12 个                   |
| MLP 隐藏维度      | 3072 (4× hidden)        |
| 总参数量          | ~86M                    |
| VRAM 需求 (bs=32) | ~4-5 GB  4080可轻松运行 |

每个 Encoder 层包含：LayerNorm → Multi-Head Attention（+残差）→ LayerNorm → MLP（+残差）。

最终取 [CLS] token 的输出接全连接层分类。

## 模型训练
---

### model.py — 使用 timm 加载 ViT-B/16

```python
import timm
import torch.nn as nn

def build_vit(num_classes=10, pretrained=True):
    # 从头训练: pretrained=False
    # 迁移学习: pretrained=True (推荐)
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model

# 或手动替换分类头
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(768, num_classes)  # 768 = hidden_dim
```

### dataset.py — CIFAR-10 示例数据集

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(data_dir='./data', batch_size=64):
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True,
                                download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(data_dir, train=False,
                                download=True, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2,
                              shuffle=False, num_workers=4,
                              pin_memory=True)
    return train_loader, val_loader
```
### train.py — 完整训练脚本

```python
import torch, timm, os
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from dataset import get_loaders

# ── 配置 ────────────────────────────────────
CFG = {
    "num_classes": 10,
    "epochs": 30,
    "batch_size": 64,    # RTX 4080 可用 128
    "lr": 1e-4,           # 微调用小学习率
    "weight_decay": 0.05,
    "warmup_epochs": 5,
    "ckpt_dir": "./checkpoints",
}
os.makedirs(CFG["ckpt_dir"], exist_ok=True)
device = torch.device("cuda")

# ── 模型 ────────────────────────────────────
model = timm.create_model('vit_base_patch16_224',
                          pretrained=True,
                          num_classes=CFG["num_classes"])
model = model.to(device)

# ── 优化器: AdamW + Cosine LR ────────────────
optimizer = optim.AdamW(model.parameters(),
                        lr=CFG["lr"],
                        weight_decay=CFG["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG["epochs"])
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()             # 混合精度
writer = SummaryWriter("./logs")

# ── 训练循环 ────────────────────────────────
train_loader, val_loader = get_loaders(
    batch_size=CFG["batch_size"])

def train_epoch(epoch):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():              # FP16 加速
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
    acc = correct / len(train_loader.dataset)
    writer.add_scalar("Loss/train", total_loss/len(train_loader), epoch)
    writer.add_scalar("Acc/train", acc, epoch)
    return total_loss/len(train_loader), acc

def val_epoch(epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
    acc = correct / len(val_loader.dataset)
    writer.add_scalar("Acc/val", acc, epoch)
    return acc

best_acc = 0
for epoch in range(CFG["epochs"]):
    tr_loss, tr_acc = train_epoch(epoch)
    val_acc = val_epoch(epoch)
    scheduler.step()
    print(f"Epoch {epoch+1:02d} | Loss {tr_loss:.4f} | TrainAcc {tr_acc:.4f} | ValAcc {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),
            f"{CFG['ckpt_dir']}/best_vit.pth")

writer.close()
print(f"Best Val Acc: {best_acc:.4f}")
```

### 启动训练与监控

```bash
# 启动训练
python train.py

# 新终端启动 TensorBoard
tensorboard --logdir=./logs --port=6006
# 浏览器打开: http://localhost:6006

# 监控 GPU 使用率
watch -n 1 nvidia-smi
```

## 推理部署
---

### infer.py — 单张图像推理

```python
import torch, timm
from PIL import Image
from torchvision import transforms

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

def load_model(ckpt_path, num_classes=10):
    model = timm.create_model('vit_base_patch16_224',
                              pretrained=False,
                              num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path,
                          map_location='cuda'))
    model.eval().cuda()
    return model

def predict(model, img_path):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    x = tf(img).unsqueeze(0).cuda()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
    top5 = probs.topk(5)
    for prob, idx in zip(top5.values, top5.indices):
        print(f"{CLASSES[idx]:15s}  {prob*100:.2f}%")

model = load_model("./checkpoints/best_vit.pth")
predict(model, "test_image.jpg")
```

### 批量推理 & 导出 ONNX

```python
# 导出 ONNX 格式 (可用于 TensorRT 加速)
dummy = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(
    model, dummy,
    "vit_base.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}},
    opset_version=17
)
print("ONNX 导出成功: vit_base.onnx")
```

### 使用 HuggingFace 预训练权重

```python
from transformers import ViTForImageClassification, ViTFeatureExtractor

# 自动下载 ImageNet-21k 预训练权重
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)
extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224"
)

img = Image.open("test.jpg")
inputs = extractor(images=img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
pred = outputs.logits.argmax(-1)
print(model.config.id2label[pred.item()])
```

## 模型转换& BenchMark测试
---

### 模型导出脚本--从best_vit.pth到onnx模型

```python
"""
ViT-Base ONNX Export Script
支持精度: FP32 / FP16 / INT8 (静态量化) / UINT8
用法:
    python export_onnx.py --checkpoint checkpoints/best_vit.pth --precision fp32
    python export_onnx.py --checkpoint checkpoints/best_vit.pth --precision fp16
    python export_onnx.py --checkpoint checkpoints/best_vit.pth --precision int8 --calib-dir ./data
    python export_onnx.py --checkpoint checkpoints/best_vit.pth --precision all
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import timm
import onnx
import onnxruntime as ort
from pathlib import Path


# ─── 颜色输出 ──────────────────────────────────────────────────────────────────
class C:
    BLUE   = "\033[94m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def log(msg, color=C.RESET):   print(f"{color}{msg}{C.RESET}")
def ok(msg):                   log(f"  ✓  {msg}", C.GREEN)
def info(msg):                 log(f"  ─  {msg}", C.BLUE)
def warn(msg):                 log(f"  ⚠  {msg}", C.YELLOW)
def err(msg):                  log(f"  ✗  {msg}", C.RED)
def title(msg):                log(f"\n{C.BOLD}{'─'*55}\n  {msg}\n{'─'*55}{C.RESET}")


# ─── 模型加载 ──────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, num_classes: int, device: str = "cpu") -> nn.Module:
    """加载 ViT-Base/16 并恢复权重"""
    title("加载模型权重")
    info(f"checkpoint : {ckpt_path}")
    info(f"num_classes: {num_classes}")

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # 兼容 DataParallel 保存的权重 (key 带 "module.")
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    ok(f"权重加载完成 → device={device}")
    return model.to(device)


# ─── FP32 导出 ─────────────────────────────────────────────────────────────────
def export_fp32(model: nn.Module, output_path: str, batch_size: int,
                input_h: int, input_w: int, opset: int) -> str:
    title("导出 FP32 ONNX")
    dummy = torch.randn(batch_size, 3, input_h, input_w)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # 验证模型结构
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    ok(f"FP32 ONNX 已保存: {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ─── FP16 导出 ─────────────────────────────────────────────────────────────────
def export_fp16(fp32_path: str, output_path: str) -> str:
    title("转换 FP16 ONNX")
    try:
        from onnxconverter_common import float16
    except ImportError:
        err("缺少依赖: pip install onnxconverter-common")
        return None

    fp32_model = onnx.load(fp32_path)
    fp16_model = float16.convert_float_to_float16(
        fp32_model,
        keep_io_types=True,   # 保持输入输出为 FP32，内部计算用 FP16
        disable_shape_infer=False,
    )
    onnx.save(fp16_model, output_path)
    onnx.checker.check_model(fp16_model)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    ok(f"FP16 ONNX 已保存: {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ─── INT8 静态量化 ─────────────────────────────────────────────────────────────
def build_calibration_dataloader(calib_dir: str, num_samples: int = 200,
                                  batch_size: int = 8, img_size: int = 224):
    """构建校准数据集 DataLoader，支持 ImageFolder 或随机数据"""
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader, Subset

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    if calib_dir and Path(calib_dir).exists():
        # 尝试作为 ImageFolder 加载
        try:
            ds = datasets.ImageFolder(calib_dir, transform=tf)
            indices = list(range(min(num_samples, len(ds))))
            ds = Subset(ds, indices)
            info(f"校准数据集: ImageFolder  {len(ds)} 张图像  ({calib_dir})")
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        except Exception:
            pass

        # 尝试 CIFAR-10 (train=False)
        try:
            ds = datasets.CIFAR10(calib_dir, train=False, download=False, transform=tf)
            indices = list(range(min(num_samples, len(ds))))
            ds = Subset(ds, indices)
            info(f"校准数据集: CIFAR-10 val  {len(ds)} 张图像")
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        except Exception:
            pass

    # Fallback: 随机噪声数据（不影响精度测试，但量化精度会低于真实校准）
    warn("未找到有效校准数据目录，使用随机数据（量化精度仅供测试，建议提供 --calib-dir）")
    data = torch.randn(num_samples, 3, img_size, img_size)
    dataset = torch.utils.data.TensorDataset(data, torch.zeros(num_samples, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def export_int8(fp32_path: str, output_path: str, calib_dir: str,
                num_calib_samples: int = 200) -> str:
    title("导出 INT8 静态量化 ONNX")
    try:
        from onnxruntime.quantization import (
            quantize_static, CalibrationDataReader,
            QuantType, QuantFormat,
            CalibrationMethod,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        err("缺少依赖: pip install onnxruntime  (需要 >=1.14)")
        return None

    # ── Step 1: 预处理（shape inference + constant folding）
    prep_path = output_path.replace(".onnx", "_prep.onnx")
    info("Step 1/3: 图形预处理 (shape inference)...")
    quant_pre_process(fp32_path, prep_path, skip_optimization=False)

    # ── Step 2: 构建 CalibrationDataReader
    loader = build_calibration_dataloader(calib_dir, num_samples=num_calib_samples)

    class VitCalibReader(CalibrationDataReader):
        def __init__(self, dataloader):
            self.dl   = dataloader
            self.iter = iter(dataloader)
            self.done = False

        def get_next(self):
            if self.done:
                return None
            try:
                imgs, _ = next(self.iter)
                return {"image": imgs.numpy().astype(np.float32)}
            except StopIteration:
                self.done = True
                return None

    # ── Step 3: 静态量化
    info("Step 2/3: 运行校准 (MinMax)...")
    info("Step 3/3: 量化模型...")
    quantize_static(
        model_input=prep_path,
        model_output=output_path,
        calibration_data_reader=VitCalibReader(loader),
        quant_format=QuantFormat.QDQ,          # QDQ 格式，兼容性最好
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,                       # 逐通道量化，精度更高
        reduce_range=False,                     # True 可解决某些硬件溢出
        extra_options={
            "EnableSubgraph": False,            # Transformer 层不拆子图
            "ForceQuantizeNoInputCheck": False,
        },
    )

    # 清理临时文件
    if os.path.exists(prep_path):
        os.remove(prep_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    ok(f"INT8 ONNX 已保存: {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ─── UINT8 动态量化（无需校准数据）─────────────────────────────────────────────
def export_dynamic_int8(fp32_path: str, output_path: str) -> str:
    title("导出 Dynamic INT8 ONNX (无需校准数据)")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        err("缺少依赖: pip install onnxruntime")
        return None

    quantize_dynamic(
        model_input=fp32_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        # optimize_model 已在新版 onnxruntime 中移除
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    ok(f"Dynamic INT8 ONNX 已保存: {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ─── 精度 & 速度基准测试 ────────────────────────────────────────────────────────
def benchmark(onnx_paths: dict, input_shape=(1, 3, 224, 224),
              num_warmup=10, num_runs=50):
    title("基准测试 (ORT CPU)")
    results = {}

    for label, path in onnx_paths.items():
        if path is None or not os.path.exists(path):
            continue
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
            sess = ort.InferenceSession(
                path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            dummy = np.random.randn(*input_shape).astype(np.float32)

            # Warmup
            for _ in range(num_warmup):
                sess.run(None, {"image": dummy})

            # Timing
            t0 = time.perf_counter()
            for _ in range(num_runs):
                out = sess.run(None, {"image": dummy})
            elapsed = (time.perf_counter() - t0) / num_runs * 1000  # ms

            size_mb = os.path.getsize(path) / 1024 / 1024
            results[label] = {"latency_ms": elapsed, "size_mb": size_mb}
            ok(f"{label:20s}  {elapsed:6.2f} ms/img   {size_mb:6.1f} MB")

        except Exception as e:
            warn(f"{label}: 基准测试跳过 ({e})")

    return results


# ─── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ViT-Base ONNX Export & Quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_vit.pth",
                        help="PyTorch 权重文件路径")
    parser.add_argument("--output-dir",  type=str, default="onnx_exports",
                        help="ONNX 输出目录")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="分类类别数")
    parser.add_argument("--input-size",  type=int, default=224,
                        help="输入图像尺寸 (H=W)")
    parser.add_argument("--batch-size",  type=int, default=1,
                        help="ONNX 导出时的静态 batch (动态 batch 通过 dynamic_axes 支持)")
    parser.add_argument("--opset",       type=int, default=17,
                        help="ONNX opset 版本")
    parser.add_argument("--precision",   type=str, default="fp32",
                        choices=["fp32", "fp16", "int8", "dynamic_int8", "all"],
                        help="导出精度: fp32 | fp16 | int8 | dynamic_int8 | all")
    parser.add_argument("--calib-dir",   type=str, default="./data",
                        help="INT8 静态量化校准数据目录 (ImageFolder 或 CIFAR-10)")
    parser.add_argument("--calib-samples", type=int, default=200,
                        help="校准样本数量")
    parser.add_argument("--benchmark",   action="store_true",
                        help="导出后运行速度基准测试")
    args = parser.parse_args()

    # ── 检查依赖 ──────────────────────────────────────────────────────────────
    title("依赖检查")
    try:
        import onnx; ok(f"onnx              {onnx.__version__}")
    except ImportError:
        err("缺少 onnx: pip install onnx"); return

    try:
        import onnxruntime; ok(f"onnxruntime       {onnxruntime.__version__}")
    except ImportError:
        err("缺少 onnxruntime: pip install onnxruntime"); return

    try:
        import onnxconverter_common; ok(f"onnxconverter_common  {onnxconverter_common.__version__}")
    except ImportError:
        if args.precision in ("fp16", "all"):
            warn("onnxconverter-common 未安装，FP16 导出将跳过")
            warn("安装命令: pip install onnxconverter-common")

    # ── 准备输出目录 ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info(f"输出目录: {out_dir.resolve()}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, args.num_classes)

    exported = {}
    fp32_path = str(out_dir / "vit_base_fp32.onnx")

    # ── FP32（所有精度都需要先导出 FP32 作为基底）─────────────────────────────
    exported["FP32"] = export_fp32(
        model, fp32_path,
        args.batch_size, args.input_size, args.input_size,
        args.opset,
    )

    # ── FP16 ──────────────────────────────────────────────────────────────────
    if args.precision in ("fp16", "all"):
        exported["FP16"] = export_fp16(
            fp32_path,
            str(out_dir / "vit_base_fp16.onnx"),
        )

    # ── INT8 静态量化 ─────────────────────────────────────────────────────────
    if args.precision in ("int8", "all"):
        exported["INT8 (static)"] = export_int8(
            fp32_path,
            str(out_dir / "vit_base_int8_static.onnx"),
            args.calib_dir,
            args.calib_samples,
        )

    # ── Dynamic INT8（无需校准）───────────────────────────────────────────────
    if args.precision in ("dynamic_int8", "all"):
        exported["INT8 (dynamic)"] = export_dynamic_int8(
            fp32_path,
            str(out_dir / "vit_base_int8_dynamic.onnx"),
        )

    # ── 基准测试 ──────────────────────────────────────────────────────────────
    if args.benchmark:
        benchmark(
            exported,
            input_shape=(1, 3, args.input_size, args.input_size),
        )

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    title("导出汇总")
    for label, path in exported.items():
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            ok(f"{label:20s}  →  {path}  ({size_mb:.1f} MB)")
        else:
            warn(f"{label:20s}  →  跳过或失败")

    log(f"\n{C.BOLD}全部完成！输出目录: {out_dir.resolve()}{C.RESET}", C.GREEN)
    log("""
使用 OnnxRuntime 加载示例:
    import onnxruntime as ort, numpy as np
    sess = ort.InferenceSession("onnx_exports/vit_base_fp32.onnx",
                                providers=["CUDAExecutionProvider",
                                           "CPUExecutionProvider"])
    out = sess.run(None, {"image": np.random.randn(1,3,224,224).astype("float32")})
    print("logits shape:", out[0].shape)
""", C.BLUE)


if __name__ == "__main__":
    main()
```

**使用方法**

```bash
# 仅 FP32（基础导出）
python export_onnx.py --checkpoint checkpoints/best_vit.pth --precision fp32

# FP16（约减少 50% 体积，RTX 4080 推理更快）
python export_onnx.py --checkpoint checkpoints/best_vit.pth --precision fp16

# INT8 静态量化（需要校准数据，精度最高的量化方式）
python export_onnx.py --checkpoint checkpoints/best_vit.pth --precision int8 --calib-dir ./data

# INT8 动态量化（无需校准数据，开箱即用）
python export_onnx.py --checkpoint checkpoints/best_vit.pth --precision dynamic_int8

# 一次性导出全部精度 + 基准测试
python export_onnx.py --checkpoint checkpoints/best_vit.pth --precision all --benchmark
```

**精度说明：**

| 精度      | 体积    | 速度   | 精度损失  | 是否需要校准数据 |
| --------- | ------- | ------ | --------- | ---------------- |
| FP32      | ~330 MB | 基准   | 无        | 否               |
| FP16      | ~165 MB | 1.5–2× | 极小      | 否               |
| INT8 静态 | ~85 MB  | 2–4×   | 小（<1%） | ✓ 需要           |
| INT8 动态 | ~85 MB  | 1.5–2× | 中等      | 否               |

几个注意点：

- **INT8 静态量化**的 `--calib-dir` 指向你的 `data/` 目录即可，脚本会自动识别 CIFAR-10 格式或 ImageFolder 格式，找不到则回退到随机数据（精度会打折）

- FP16 在 RTX 4080 的 Tensor Core 上效果最佳，实际推理速度收益最大

- 脚本会自动清理量化过程中的临时文件，并输出每个模型的大小和延迟汇总
  
### 导出结果展示

```bash
(vit_env) shi19@AIPC:~/ViTBase$ ls onnx_exports/
vit_base_fp16.onnx  vit_base_fp32.onnx  vit_base_int8_dynamic.onnx  vit_base_int8_static.onnx
(vit_env) shi19@AIPC:~/ViTBase$
```

### 性能测试benchmark

```python
"""
ViT-Base ONNX 批量检测性能测试脚本
数据集: CIFAR-10 (自动下载) 或 ImageNet-val (本地)
测试项:
  - 各精度模型 Top-1 / Top-5 准确率
  - 不同 batch size 下的吞吐量 (images/sec) 与延迟 (ms/img)
  - 模型体积对比
  - 输出详细对比表 + CSV 报告

用法:
    python benchmark_onnx.py
    python benchmark_onnx.py --dataset imagenet --data-dir /path/to/imagenet/val
    python benchmark_onnx.py --batch-sizes 1 8 16 32 --num-samples 1000
    python benchmark_onnx.py --providers cuda cpu   # 指定 ORT 执行后端
"""

import argparse
import csv
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("请先安装: pip install onnxruntime-gpu  或  pip install onnxruntime")

# ─── 终端颜色 ──────────────────────────────────────────────────────────────────
class C:
    BLUE = "\033[94m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    RED  = "\033[91m"; BOLD  = "\033[1m";  RESET  = "\033[0m"
    CYAN = "\033[96m"

def title(s): print(f"\n{C.BOLD}{'═'*60}\n  {s}\n{'═'*60}{C.RESET}")
def ok(s):    print(f"  {C.GREEN}✓{C.RESET}  {s}")
def info(s):  print(f"  {C.BLUE}─{C.RESET}  {s}")
def warn(s):  print(f"  {C.YELLOW}⚠{C.RESET}  {s}")
def err(s):   print(f"  {C.RED}✗{C.RESET}  {s}")
def row(s):   print(f"  {s}")


# ─── CIFAR-10 类名 ─────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─── 数据集构建 ────────────────────────────────────────────────────────────────
def build_dataset(dataset_name: str, data_dir: str, img_size: int = 224,
                  num_samples: int = None):
    """
    返回 (dataset, class_names)
    支持: cifar10 | imagenet
    """
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    title(f"加载数据集: {dataset_name.upper()}")

    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=val_tf,
        )
        class_names = CIFAR10_CLASSES
        ok(f"CIFAR-10 测试集  {len(ds)} 张  (已自动下载至 {data_dir})")

    elif dataset_name == "imagenet":
        val_path = Path(data_dir)
        if not val_path.exists():
            raise SystemExit(f"ImageNet val 目录不存在: {val_path}")
        ds = datasets.ImageFolder(val_path, transform=val_tf)
        class_names = ds.classes
        ok(f"ImageNet val  {len(ds)} 张  {len(class_names)} 类")

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    if num_samples and num_samples < len(ds):
        indices = list(range(num_samples))
        ds = Subset(ds, indices)
        info(f"截取前 {num_samples} 张用于测试")

    return ds, class_names


# ─── ORT Session 工厂 ──────────────────────────────────────────────────────────
def build_session(onnx_path: str, providers: list) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = min(8, os.cpu_count() or 4)

    available = ort.get_available_providers()
    valid_providers = [p for p in providers if p in available]
    if not valid_providers:
        valid_providers = ["CPUExecutionProvider"]
        warn(f"请求的 provider 均不可用，回退到 CPU")

    sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=valid_providers)
    active = sess.get_providers()
    return sess, active


# ─── 准确率评估 ────────────────────────────────────────────────────────────────
def evaluate_accuracy(sess: ort.InferenceSession, loader: DataLoader,
                      label: str) -> dict:
    """计算 Top-1 / Top-5 准确率"""
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    top1_correct = top5_correct = total = 0

    print(f"\n  [{label}] 准确率评估...", end="", flush=True)
    t_start = time.perf_counter()

    for imgs, labels in loader:
        imgs_np = imgs.numpy().astype(np.float32)
        logits  = sess.run([output_name], {input_name: imgs_np})[0]  # (B, C)

        # Top-1
        preds_top1 = np.argmax(logits, axis=1)
        top1_correct += (preds_top1 == labels.numpy()).sum()

        # Top-5
        top5_preds = np.argsort(logits, axis=1)[:, -5:]
        for i, lbl in enumerate(labels.numpy()):
            if lbl in top5_preds[i]:
                top5_correct += 1

        total += len(labels)
        print(".", end="", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f" 完成 ({elapsed:.1f}s)")

    return {
        "top1": top1_correct / total * 100,
        "top5": top5_correct / total * 100,
        "total": total,
    }


# ─── 吞吐量 & 延迟测试 ─────────────────────────────────────────────────────────
def benchmark_throughput(sess: ort.InferenceSession, batch_size: int,
                          img_size: int, num_warmup: int = 20,
                          num_runs: int = 100) -> dict:
    """
    测量指定 batch_size 下的:
      - 平均延迟 (ms/batch)
      - 单张延迟 (ms/img)
      - 吞吐量 (images/sec)
      - P50 / P95 / P99 延迟
    """
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    dummy = np.random.randn(batch_size, 3, img_size, img_size).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        sess.run([output_name], {input_name: dummy})

    # Timing
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sess.run([output_name], {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    lat_arr = np.array(latencies)
    mean_lat = lat_arr.mean()
    return {
        "batch_size":   batch_size,
        "mean_ms":      mean_lat,
        "per_img_ms":   mean_lat / batch_size,
        "throughput":   batch_size / (mean_lat / 1000),
        "p50_ms":       np.percentile(lat_arr, 50),
        "p95_ms":       np.percentile(lat_arr, 95),
        "p99_ms":       np.percentile(lat_arr, 99),
    }


# ─── 表格打印工具 ──────────────────────────────────────────────────────────────
def print_table(headers: list, rows: list, col_widths: list = None):
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in ([headers] + rows))
                      for i in range(len(headers))]
        col_widths = [max(w, 6) for w in col_widths]

    sep = "─" * (sum(col_widths) + 3 * len(col_widths) + 1)
    fmt = "  │ " + " │ ".join(f"{{:<{w}}}" for w in col_widths) + " │"

    print(f"  {sep}")
    print(fmt.format(*headers))
    print(f"  {sep}")
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))
    print(f"  {sep}")


# ─── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ViT-Base ONNX 批量性能测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-dir",    type=str, default="onnx_exports",
                        help="ONNX 模型目录")
    parser.add_argument("--models",      type=str, nargs="+",
                        default=["fp32", "fp16", "int8_static", "int8_dynamic"],
                        help="要测试的模型后缀列表")
    parser.add_argument("--dataset",     type=str, default="cifar10",
                        choices=["cifar10", "imagenet"],
                        help="评估数据集")
    parser.add_argument("--data-dir",    type=str, default="./data",
                        help="数据集根目录")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="参与准确率评估的样本数 (None=全量)")
    parser.add_argument("--acc-batch",   type=int, default=32,
                        help="准确率评估时的 batch size")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 4, 8, 16, 32],
                        help="吞吐量测试的 batch size 列表")
    parser.add_argument("--img-size",    type=int, default=224,
                        help="输入图像尺寸")
    parser.add_argument("--num-warmup",  type=int, default=20,
                        help="吞吐量测试预热次数")
    parser.add_argument("--num-runs",    type=int, default=100,
                        help="吞吐量测试正式运行次数")
    parser.add_argument("--providers",   type=str, nargs="+",
                        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
                        help="OnnxRuntime 执行 provider")
    parser.add_argument("--output-csv",  type=str, default="benchmark_results.csv",
                        help="CSV 报告保存路径")
    parser.add_argument("--skip-acc",    action="store_true",
                        help="跳过准确率评估，仅测速度")
    args = parser.parse_args()

    # ── 发现模型文件 ──────────────────────────────────────────────────────────
    title("发现 ONNX 模型")
    onnx_dir = Path(args.onnx_dir)
    model_map = {}  # label → path

    suffix_to_label = {
        "fp32":         "FP32",
        "fp16":         "FP16",
        "int8_static":  "INT8-Static",
        "int8_dynamic": "INT8-Dynamic",
    }
    for suffix in args.models:
        candidate = onnx_dir / f"vit_base_{suffix}.onnx"
        if candidate.exists():
            label = suffix_to_label.get(suffix, suffix.upper())
            model_map[label] = str(candidate)
            size_mb = candidate.stat().st_size / 1024 / 1024
            ok(f"{label:15s}  {candidate.name}  ({size_mb:.1f} MB)")
        else:
            warn(f"找不到: {candidate}，跳过")

    if not model_map:
        raise SystemExit(f"在 {onnx_dir} 下未找到任何 ONNX 模型")

    # ── 检查 ORT Provider ────────────────────────────────────────────────────
    title("OnnxRuntime 环境")
    info(f"ORT 版本     : {ort.__version__}")
    info(f"可用 provider: {ort.get_available_providers()}")
    info(f"请求 provider: {args.providers}")

    # ── 加载数据集 ────────────────────────────────────────────────────────────
    dataset, class_names = build_dataset(
        args.dataset, args.data_dir, args.img_size, args.num_samples
    )
    acc_loader = DataLoader(
        dataset, batch_size=args.acc_batch,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    info(f"类别数: {len(class_names)}")

    # ══════════════════════════════════════════════════════════════════════════
    # 第一阶段：准确率评估
    # ══════════════════════════════════════════════════════════════════════════
    acc_results = {}  # label → {top1, top5}

    if not args.skip_acc:
        title("阶段 1/2：准确率评估")
        for label, path in model_map.items():
            try:
                sess, active = build_session(path, args.providers)
                info(f"{label}  使用 provider: {active[0]}")
                res = evaluate_accuracy(sess, acc_loader, label)
                acc_results[label] = res
                ok(f"{label:15s}  Top-1: {res['top1']:.2f}%   Top-5: {res['top5']:.2f}%   "
                   f"样本数: {res['total']}")
            except Exception as e:
                err(f"{label} 准确率评估失败: {e}")
                acc_results[label] = {"top1": float("nan"), "top5": float("nan"), "total": 0}
    else:
        info("已跳过准确率评估 (--skip-acc)")

    # ══════════════════════════════════════════════════════════════════════════
    # 第二阶段：吞吐量 & 延迟测试
    # ══════════════════════════════════════════════════════════════════════════
    title("阶段 2/2：吞吐量 & 延迟测试")
    perf_results = {}  # label → [batch_result, ...]

    for label, path in model_map.items():
        perf_results[label] = []
        try:
            sess, active = build_session(path, args.providers)
            info(f"\n  {C.BOLD}{label}{C.RESET}  [{active[0]}]")
            for bs in args.batch_sizes:
                r = benchmark_throughput(
                    sess, bs, args.img_size,
                    num_warmup=args.num_warmup,
                    num_runs=args.num_runs,
                )
                perf_results[label].append(r)
                row(f"    batch={bs:3d}  "
                    f"latency={r['mean_ms']:7.2f} ms  "
                    f"per_img={r['per_img_ms']:6.3f} ms  "
                    f"throughput={r['throughput']:8.1f} img/s  "
                    f"p95={r['p95_ms']:7.2f} ms")
        except Exception as e:
            err(f"{label} 性能测试失败: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # 汇总报告
    # ══════════════════════════════════════════════════════════════════════════
    title("汇总报告")

    # ── 准确率汇总表 ──────────────────────────────────────────────────────────
    if acc_results:
        print(f"\n  {C.BOLD}【准确率对比】{C.RESET}")
        fp32_top1 = acc_results.get("FP32", {}).get("top1", float("nan"))

        acc_headers = ["模型", "体积(MB)", "Top-1(%)", "Top-5(%)", "vs FP32"]
        acc_rows = []
        for label, path in model_map.items():
            size_mb = os.path.getsize(path) / 1024 / 1024
            res = acc_results.get(label, {})
            top1 = res.get("top1", float("nan"))
            top5 = res.get("top5", float("nan"))
            delta = f"{top1 - fp32_top1:+.2f}%" if not np.isnan(top1) and label != "FP32" else "─"
            acc_rows.append([
                label,
                f"{size_mb:.1f}",
                f"{top1:.2f}" if not np.isnan(top1) else "N/A",
                f"{top5:.2f}" if not np.isnan(top5) else "N/A",
                delta,
            ])
        print_table(acc_headers, acc_rows, [14, 10, 10, 10, 10])

    # ── 吞吐量汇总表（以 batch=16 为代表）────────────────────────────────────
    target_bs = 16
    print(f"\n  {C.BOLD}【吞吐量对比 (batch={target_bs})】{C.RESET}")
    thr_headers = ["模型", "吞吐(img/s)", "延迟(ms/img)", "P95(ms/batch)", "加速比"]
    thr_rows = []

    fp32_thr = None
    for label in model_map:
        for r in perf_results.get(label, []):
            if r["batch_size"] == target_bs and label == "FP32":
                fp32_thr = r["throughput"]

    for label in model_map:
        for r in perf_results.get(label, []):
            if r["batch_size"] == target_bs:
                speedup = f"{r['throughput']/fp32_thr:.2f}×" if fp32_thr else "─"
                thr_rows.append([
                    label,
                    f"{r['throughput']:.1f}",
                    f"{r['per_img_ms']:.3f}",
                    f"{r['p95_ms']:.2f}",
                    speedup,
                ])
    if thr_rows:
        print_table(thr_headers, thr_rows, [14, 12, 14, 14, 8])
    else:
        warn(f"未找到 batch={target_bs} 的测试结果")

    # ── 各 batch size 吞吐量矩阵 ─────────────────────────────────────────────
    print(f"\n  {C.BOLD}【吞吐量矩阵 (img/s)】{C.RESET}")
    matrix_headers = ["模型"] + [f"bs={bs}" for bs in args.batch_sizes]
    matrix_rows = []
    for label in model_map:
        results_by_bs = {r["batch_size"]: r for r in perf_results.get(label, [])}
        row_data = [label]
        for bs in args.batch_sizes:
            r = results_by_bs.get(bs)
            row_data.append(f"{r['throughput']:.0f}" if r else "─")
        matrix_rows.append(row_data)
    if matrix_rows:
        col_w = [14] + [10] * len(args.batch_sizes)
        print_table(matrix_headers, matrix_rows, col_w)

    # ── CSV 报告 ──────────────────────────────────────────────────────────────
    title("保存 CSV 报告")
    csv_path = args.output_csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 准确率
        writer.writerow(["=== 准确率 ==="])
        writer.writerow(["model", "size_mb", "top1_pct", "top5_pct", "num_samples"])
        for label, path in model_map.items():
            size_mb = os.path.getsize(path) / 1024 / 1024
            res = acc_results.get(label, {})
            writer.writerow([
                label, f"{size_mb:.2f}",
                f"{res.get('top1', ''):.4f}" if res.get('top1') else "",
                f"{res.get('top5', ''):.4f}" if res.get('top5') else "",
                res.get("total", ""),
            ])
        writer.writerow([])
        # 吞吐量
        writer.writerow(["=== 吞吐量 & 延迟 ==="])
        writer.writerow(["model", "batch_size", "mean_lat_ms", "per_img_ms",
                          "throughput_img_s", "p50_ms", "p95_ms", "p99_ms"])
        for label in model_map:
            for r in perf_results.get(label, []):
                writer.writerow([
                    label, r["batch_size"],
                    f"{r['mean_ms']:.4f}", f"{r['per_img_ms']:.4f}",
                    f"{r['throughput']:.2f}",
                    f"{r['p50_ms']:.4f}", f"{r['p95_ms']:.4f}", f"{r['p99_ms']:.4f}",
                ])

    ok(f"CSV 报告已保存: {csv_path}")
    print(f"\n{C.BOLD}{C.GREEN}  全部测试完成！{C.RESET}\n")


if __name__ == "__main__":
    main()
```

#### **使用方法**

**直接运行（CIFAR-10 会自动下载）：**

```bash
python benchmark_onnx.py
```

**常用参数示例：**

```bash
# 快速测试，只取 500 张，跳过准确率
python benchmark_onnx.py --num-samples 500 --skip-acc

# 完整评估 + 自定义 batch size 组合
python benchmark_onnx.py --batch-sizes 1 8 32 64 --num-samples 5000

# 指定 GPU 执行（需要 onnxruntime-gpu）
python benchmark_onnx.py --providers CUDAExecutionProvider CPUExecutionProvider

# 使用本地 ImageNet val 集
python benchmark_onnx.py --dataset imagenet --data-dir /path/to/imagenet/val
```

**脚本会输出三张汇总表 + CSV 报告：**

| 测试项      | 内容                                              |
| ----------- | ------------------------------------------------- |
| 准确率对比  | Top-1 / Top-5，以及各量化模型相对 FP32 的精度损失 |
| 吞吐量对比  | 以 batch=16 为代表，显示加速比                    |
| 吞吐量矩阵  | 所有模型 × 所有 batch size 的 img/s 全览          |
| P50/P95/P99 | 延迟分位数，反映稳定性                            |
| CSV 报告    | `benchmark_results.csv`，方便后续分析             |

## 调优建议
---

针对 RTX 4080 Laptop (12GB) + WSL2 环境的专项优化建议：

<img width="544" height="90" alt="image" src="https://github.com/user-attachments/assets/4a18c70b-367a-4512-976b-9c387540420f" />

### 推荐配置 (RTX 4080 Laptop)

| batch_size      | 64 ~ 128（FP16模式）  |
| --------------- | --------------------- |
| 精度            | AMP（torch.cuda.amp） |
| 优化器          | AdamW，lr=1e-4        |
| 调度器          | Cosine Annealing      |
| Warmup          | 5 epochs              |
| Label Smoothing | 0.1                   |

### 梯度累积（模拟大 batch）

```python
# 每 4 步更新一次 = 等效 batch=256
ACCUM_STEPS = 4
optimizer.zero_grad()
for i, (imgs, labels) in enumerate(train_loader):
    with autocast():
        loss = criterion(model(imgs.cuda()), labels.cuda())
        loss = loss / ACCUM_STEPS
    scaler.scale(loss).backward()
    if (i + 1) % ACCUM_STEPS == 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### WSL2 内存限制配置

```python
# Windows 端: %USERPROFILE%\.wslconfig
[wsl2]
memory=16GB      # 按实际内存调整
processors=8
gpuMemoryFraction=0.9  # GPU 内存分配比例

# 修改后重启 WSL2:
wsl --shutdown
```

### 常见问题排查

- OOM (显存溢出)

减小 batch_size，或启用 torch.backends.cudnn.benchmark = True，或使用梯度检查点 model.gradient_checkpointing_enable()

- WSL2 GPU 不可见

确保 Windows 侧安装了支持 WSL2-GPU 的驱动（581.83 ✓），在 WSL2 中不要另装 CUDA Toolkit，直接用 PyTorch 的 CUDA 版本即可。

- DataLoader 卡住

WSL2 下 num_workers > 0 可能报错，改为 num_workers=0 或设置 multiprocessing_context='fork'

