## Yolo11 OBB 

---

YOLO11 的 OBB（Oriented Bounding Box，旋转/定向边界框）确实专门用于检测带有倾斜角度、非轴对齐的物体。

相比于传统目标检测使用的水平矩形框（AABB），OBB 通过引入**旋转角度参数**，能够更紧密、精准地贴合目标轮廓。

以下是对 YOLO11 OBB 技术的核心解析：

### **🎯 传统水平框的痛点**

在无人机航拍、遥感影像或工业质检等场景中，物体往往呈现各种倾斜姿态。如果使用传统的水平框（AABB）：

- **背景噪声大**：水平框会框入大量无关的背景像素（如天空、草地），平均包含约 42.7% 的背景区域。
- **密集场景易混淆**：对于长宽比悬殊（如输电线、飞机）或密集排列（如斜向停放的车辆）的物体，水平框会产生大量重叠，导致漏检和误检。

OBB 通过输出 `(x, y, w, h, θ)` 五参数（中心点坐标、宽、高、旋转角度） ，解决这些问题，使检测框紧密贴合目标实际朝向。

### **🧠 YOLO11-OBB 的核心技术亮点**

**1. 单位圆映射解决“角度跳变”**
直接预测角度（如 0°~360°）会遇到边界不连续问题（例如 350° 和 10° 实际只差 20°，但数值差巨大，导致模型训练崩溃）。

YOLO11-OBB 创新性地将角度映射为 `(sinθ, cosθ)` 的二维坐标进行预测，再通过 `atan2` 函数恢复角度，保证角度空间的连续性，大幅提升模型的收敛速度和预测稳定性。

**2. 旋转敏感的 NMS（非极大值抑制）**
传统的 NMS 只能计算水平框的交并比（IoU）。

YOLO11-OBB 引入基于旋转 IoU（RIoU）的 NMS 变体，能够精确计算两个旋转矩形之间的重叠面积，有效去除冗余的重复检测框。

**3. 动态标签分配与复合损失函数**
在训练阶段，模型会根据目标的长宽比自动调整位置、尺寸和角度损失的权重。

对于长宽比极大的物体（如 >5:1），会赋予角度权重更高的比例，确保长条形物体的方向预测足够精准。

### **🛠️ 部署建议**

- **数据标注**：推荐使用支持旋转框标注的工具（如 Labelme 或 X-AnyLabeling），标注时需注意顶点顺序的统一（如始终顺时针），否则会导致训练时角度震荡。
- **模型导出**：在将模型导出为 ONNX 格式时，必须设置 `opset >= 12` 以支持旋转角度的输出，并建议关闭<mark>**动态尺寸（`dynamic=False`）**</mark>以避免后续转换 TensorRT 失败。
- **角度限制**：Ultralytics 官方实现的 YOLO11-OBB 角度被限制在 **0–90 度**范围内（不包括 90 度），不支持 90 度或更大的角度。

**总结**：YOLO11-OBB 通过在检测头增加角度预测分支，并配合旋转感知的损失函数与 NMS 后处理，成为了处理航拍、遥感、工业倾斜零件等场景的“杀手锏”。



## 工程实践代码

---



YOLO11-OBB 在**遥感/航拍场景**下的完整工程实践代码示例。

在实际的遥感或航拍项目中，数据集通常庞大且复杂。

以下代码涵盖了从**环境准备、自定义数据集配置、模型训练、评估验证**到**推理预测**的全流程，并加入了针对航拍场景的优化参数。

### **数据集配置 (**`dataset.yaml`**)**

YOLO 框架要求通过 YAML 文件来定义数据集路径和类别。假设航拍车辆数据集结构如下：

```yaml
# dataset.yaml
path: /path/to/your/aerial_dataset  # 数据集根目录
train: images/train                 # 训练集图像路径
val: images/val                     # 验证集图像路径

# 类别数量与名称
names:
  0: car
  1: truck
  2: bus
  3: airplane
```

*注意：标注文件（`.txt`）需与图像同名，格式为 `class x_center y_center width height angle`，其中 `angle` 为弧度制（0~90度）。*

------

### **核心训练与评估代码 (**`train_obb.py`**)**

```python
from ultralytics import YOLO
import torch

def train_aerial_obb():
    # 1. 加载预训练的 OBB 模型
    # 可选: yolo11n-obb.pt, yolo11s-obb.pt, yolo11m-obb.pt 等
    # 遥感/航拍场景建议至少使用 's' (small) 或 'm' (medium) 以捕捉小目标
    model = YOLO('yolo11s-obb.pt') 

    # 2. 开始训练
    results = model.train(
        data='dataset.yaml',       # 配置文件路径
        epochs=150,                # 航拍数据集建议训练轮数
        imgsz=1024,                # 航拍图像分辨率高，建议设为 1024 或 1536
        batch=16,                  # 根据显存调整，1024分辨率下建议 8~16
        device='0',                # 使用 GPU 0，如果是 CPU 则设为 'cpu'
        
        # --- 针对航拍场景的优化参数 ---
        degrees=45.0,              # 随机旋转增强（航拍物体角度多变，建议开启）
        mosaic=1.0,                # 开启马赛克增强，提升对小目标的鲁棒性
        close_mosaic=20,           # 最后 20 个 epoch 关闭 mosaic，稳定收敛
        mixup=0.1,                 # 混合增强，防止过拟合
        
        # --- 损失与优化器 ---
        cos_lr=True,               # 余弦退火学习率，有助于后期跳出局部最优
        optimizer='AdamW',         # 航拍小目标建议使用 AdamW 优化器
        lr0=0.001,                 # 初始学习率
        
        # --- 其他 ---
        project='runs/aerial_obb', # 实验保存目录
        name='exp_1024_adamw',     # 实验名称
        exist_ok=True              # 允许覆盖同名实验
    )
    
    print("训练完成！")
    return model

if __name__ == '__main__':
    train_aerial_obb()
```

------

### ** 模型验证与推理代码 (**`predict_obb.py`**)**

训练完成后，我们需要评估模型在验证集上的表现，并对单张航拍图像进行推理可视化。

```python
from ultralytics import YOLO
import cv2

def evaluate_and_predict():
    # 1. 加载训练好的最佳权重
    model = YOLO('runs/aerial_obb/exp_1024_adamw/weights/best.pt')

    # 2. 在验证集上评估
    metrics = model.val(
        data='dataset.yaml',
        imgsz=1024,
        split='val',               # 评估验证集
        save_json=True             # 保存 JSON 结果，方便后续分析
    )
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")

    # 3. 单张图像推理与可视化
    source_image = '/path/to/your/aerial_dataset/images/val/00001.jpg'
    
    results = model.predict(
        source=source_image,
        imgsz=1024,
        conf=0.25,                 # 置信度阈值
        iou=0.45,                  # 旋转 NMS 的 IoU 阈值
        save=True,                 # 保存带预测框的图像
        project='runs/predict',
        name='aerial_demo'
    )

    # 4. 解析结果（可选：获取具体的旋转框坐标和角度）
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            # 获取旋转框的四个顶点坐标 (N, 4, 2)
            xyxyxyxy = boxes.xyxyxyxy
            # 获取类别和置信度
            cls = boxes.cls
            conf = boxes.conf
            
            print(f"检测到 {len(boxes)} 个目标。")
            # 你可以在这里加入自定义的业务逻辑，例如过滤特定角度的车辆

if __name__ == '__main__':
    evaluate_and_predict()
```

------

### **针对航拍/遥感场景的进阶建议**

1. **切片推理（SAHI）**：
   航拍图像通常非常大（如 4000x4000），直接 resize 到 1024 会导致小目标（如远处的汽车）完全消失。

   强烈建议结合 **SAHI (Slicing Aided Hyper Inference)** 库进行切片推理：

   ```python
   from sahi import AutoDetectionModel
   from sahi.predict import get_sliced_prediction
   
   detection_model = AutoDetectionModel.from_pretrained(
       model_type='ultralytics',
       model_path='runs/aerial_obb/exp_1024_adamw/weights/best.pt',
       confidence_threshold=0.25,
       device='cuda:0'
   )
   
   result = get_sliced_prediction(
       "/path/to/large_aerial_image.jpg",
       detection_model,
       slice_height=512,
       slice_width=512,
       overlap_height_ratio=0.2,
       overlap_width_ratio=0.2
   )
   result.export_visuals(export_dir="runs/sahi_results/")
   ```

2. **角度标注的规范性**：
   在训练 OBB 模型时，如果标注工具导出的角度不一致（有的顺时针，有的逆时针），会导致模型在预测角度时产生极大的困惑（Loss 无法下降）。务必在数据预处理阶段统一角度定义。

3. **长宽比极大的目标**：
   如果你的场景包含大量长条形目标（如输电线、船只），可以在 `model.train()` 中适当调大 `angle_loss_weight`（如果底层支持自定义超参），或者在数据增强时减少 `shear`（错切）的比例，以免破坏目标的长条几何特征。

