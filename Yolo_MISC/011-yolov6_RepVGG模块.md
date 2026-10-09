## Yolov6  RepVGG结构

----

### 基础信息

**Rep 是 Re-parameterization 的缩写**，也就是“重(**重新**)参数化”。

把训练时的多分支权重重新整理、合并成推理时的单分支权重

所以 **RepVGG** 可以理解为：

> **Re-parameterized VGG**
> 或 **Reparameterizable VGG-style Network**

它不是“重复”“代表”“修复”的意思，而是指：训练时用多分支结构，推理时通过数学等价把多分支**重参数化**成单路 3×3 卷积。

| 术语                         | 含义                                             |
| :--------------------------- | :----------------------------------------------- |
| **RepVGG**                   | 重参数化 VGG 风格网络                            |
| **RepBlock / RepConv**       | YOLOv6 里借鉴 RepVGG 的重参数化模块              |
| **Re-parameterization**      | 把训练时的多分支结构等价合并成推理时的单分支结构 |
| **Reparameterization Trick** | VAE 里的“重参数化采样技巧”，和 RepVGG 不是一回事 |



**RepVGG 是一种“训练时多分支、推理时单路 3×3 卷积”的结构重参数化网络。**

训练阶段用多分支结构提高表达能力，推理阶段把这些分支等价合并成一个简单的 3×3 卷积 + ReLU 结构，在不损失精度的前提下提升推理速度。

### **训练时：多分支结构**

一个 RepVGG Block 在训练阶段通常有三条并行路径：

- **3×3 卷积分支**：主特征提取路径；
- **1×1 卷积分支**：做通道间信息交互；
- **恒等映射分支**：类似残差连接，帮助梯度回传。

输出可以写成：

<img width="500" height="78" alt="image" src="https://github.com/user-attachments/assets/a11e4ce7-dc93-4ec9-82dc-c05415c72cd6" />

这样训练时模型表达能力更强，梯度流动也更稳定。

### **推理时：合并成单路 3×3 卷积**

推理前，通过**结构重参数化**把三条分支合并：

1. 把每个分支后面的 BN 层“吸”进卷积里，变成带 bias 的卷积；
2. 把 1×1 卷积核和恒等映射对应的权重补零成 3×3 形状；
3. 三个 3×3 卷积核按位置相加，得到一个等效 3×3 卷积核；
4. 推理时就只剩一个 **3×3 Conv + ReLU**，没有 BN、没有残差、没有多分支。

### **为什么 YOLOv6 会借鉴它？**

YOLOv5 的 Backbone 和 Neck 主要用 CSP 结构，比如 C3 模块，属于多分支残差结构。

YOLOv6 为了让模型在 GPU 和边缘设备上推理更高效，把其中一部分 CSP 模块替换成了 **RepBlock / CSPStackRep Block**，也即RepVGG 风格的重参数化模块。

具体来说：

- **小模型**：Backbone 使用 RepBlock，推理时变成 RepConv；
- **大模型**：使用 CSPStackRep Block，结合 CSP 思想与 RepVGG 重参数化；
- **Neck 部分**：把原来 PAN 中的 CSP-Block 换成 RepBlock，形成 **Rep-PAN**。

### **一句话理解**

**RepVGG 不是单纯把 VGG 搬回来，而是让网络“训练时像 ResNet 一样强，推理时像 VGG 一样快”。** 

YOLOv6 借鉴它，是为了在保持检测精度的同时，降低推理延迟、提高硬件利用率。
