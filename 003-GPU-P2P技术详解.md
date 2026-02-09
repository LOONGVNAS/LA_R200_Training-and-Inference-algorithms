##  背景介绍
---

<img width="1080" height="608" alt="640" src="https://github.com/user-attachments/assets/036af077-84e8-4200-ba0d-9bd64aec6594" />

GPUDirect P2P（Peer-to-Peer）是 NVIDIA 开发的一项关键 GPU 通信技术，旨在**提升多 GPU 系统中显存间数据传输的效率**。它通过**绕过主机内存（Host Memory）和 CPU 的中转**，实现 GPU 与 GPU 之间的**直接点对点通信**，从而显著降低延迟、提高带宽，并减少系统资源开销。

**主要内容：**

- 1. 技术背景
- 2. GPUDirect P2P工作原理
- 3. GPUDirect P2P关键技术特性
- 4. GPUDirect P2P启用条件
- 5. GPUDirect P2P典型应用场景
- 6. GPUDirect P2P限制与挑战
- 7. GPUDirect P2P启用验证
- 8. 写在最后



## **1. 技术背景**

1）在没有 GPUDirect P2P 的传统架构中，若要将数据从 GPU A 传送到 GPU B，流程如下：

- `GPU A → 主机内存（通过 PCIe）`
- `主机内存 → GPU B（再次通过 PCIe）`

2）以上情况导致：

- 两次 PCIe 传输
- CPU 参与调度/拷贝
- 高延迟、低带宽、高 CPU 负载

3）GPUDirect P2P可实现：

- `GPU A`直接写入`GPU B`的显存
- 仅一次 PCIe 传输（或通过更高带宽的 NVLink）
- 无需CPU干预

## **2. GPUDirect P2P工作原理**

1）底层依赖

- PCIe 总线支持：GPU 必须连接到同一 PCIe 根复合体（Root Complex），且主板芯片组需支持 P2P。关于Root Complex，具体可以参见：[PCIe通信组件、链路训练与链路均衡过程](https://mp.weixin.qq.com/s?__biz=Mzk2NDEyMTM1Mg==&mid=2247488028&idx=1&sn=d1b394498e621f7e9756ec1874acb6bf&scene=21#wechat_redirect)
- NVIDIA 驱动与 CUDA Toolkit：CUDA 提供 `cudaDeviceEnablePeerAccess()` 等 API 启用 P2P。
- 硬件 BAR 支持：早期受限于小 BAR（Base Address Register，通常仅 256MB），现代系统普遍支持 Resizable BAR (ReBAR)，大幅提升可访问显存范围。

> **注意**：即使物理上两块 GPU 插在同一主板，若 BIOS/UEFI 或驱动未启用 P2P，仍无法使用。

2）内存地址映射

- 每个 GPU 的显存在启用 P2P 后，可被其他 GPU 通过虚拟地址直接访问。
- CUDA 运行时通过 IOMMU 或 ATS（Address Translation Services）机制完成地址转换。

> IOMMU 是 CPU MMU（Memory Management Unit）在 I/O 子系统中的延伸。它为外设（如 GPU、网卡、NVMe SSD 等）提供地址转换和内存保护机制，使得这些设备可以像 CPU 一样使用虚拟地址访问系统内存，同时保障系统的安全性与稳定性。

3）数据路径

- 若两 GPU 通过 NVLink 互联（如 A100/H100），P2P 通信走 NVLink（带宽高达 600–900 GB/s）。
- 若仅通过 PCIe（如消费级 RTX 卡），则走 PCIe P2P（PCIe 4.0 x16 ≈ 32 GB/s 双向）。

## **3. GPUDirect P2P关键技术特性**

| 特性     | 说明                                                |
| -------- | --------------------------------------------------- |
| 低延迟   | 避免主机内存中转，端到端延迟降低 30%～70%           |
| 高带宽   | 充分利用 PCIe 或 NVLink 带宽                        |
| 零拷贝   | 数据无需复制到主机内存                              |
| CPU 卸载 | CPU 不参与数据搬运，释放计算资源                    |
| 框架集成 | TensorFlow、PyTorch、MXNet 等通过 NCCL 自动利用 P2P |

> NCCL（NVIDIA Collective Communications Library） 是专为多 GPU/多节点通信优化的库，自动检测并启用 GPUDirect P2P 和 NVLink。

## **4. GPUDirect P2P启用条件**

1）启用硬件条件：

- 同一主机内的多个 NVIDIA GPU（Tesla/Quadro/Data Center 系列支持更完整）
- 主板支持 PCIe P2P（服务器主板通常默认开启，消费主板可能需 BIOS 设置）
- 推荐启用 Resizable BAR / Above 4G Decoding

2）启用软件条件：

- NVIDIA 驱动 ≥ 418（建议最新稳定版）
- CUDA Toolkit ≥ 10.0
- 操作系统支持（Linux 推荐，Windows 对 P2P 支持有限）

## **5. GPUDirect P2P典型应用场景**

1）深度学习多卡训练：AllReduce、梯度同步；

2）HPC 多 GPU 协同计算：流体仿真、分子动力学；

3）实时视频处理：多 GPU 流水线处理帧数据；

4）数据库加速：GPU 显存间直接交换中间结果。

## **6. GPUDirect P2P限制与挑战**

1）不跨 NUMA 节点：若 GPU 分属不同 CPU 插槽（不同 PCIe Root Complex），P2P 可能不可用。

2）虚拟化环境限制：在云平台（如 AWS、Azure）中，P2P 支持取决于底层虚拟化方案（如 SR-IOV + vGPU 配置）。

3）Windows 支持弱：微软 WDDM 驱动模型对 P2P 限制较多，Linux + TCC 模式更可靠。



## **7. GPUDirect P2P启用验证**

1）因为涉及到多GPU，因此比较好的方式是在云上租赁GPU服务器进行实验验证与学习。假设在8卡RTX 5090 GPU服务器（物理服务器）按需租赁4卡RTX 5090，使用`nvidia-smi topo -m`命令查看输出，具体如下：

![Pasted image 20260204201124.png](https://mmbiz.qpic.cn/sz_mmbiz_png/cCvejiaRribibSme6GJ9qQzY2tILSmC2bticfTLPXEd5QMBL9KMsQ9zrico0cNNSg1hyQaf26Wj30Hd1WtotPPWwpaQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=1)

2）输出中的含义如下：

- `X`：表示同一个 GPU（对角线）。
- `NODE`：表示这两个 GPU 位于同一个 NUMA 节点内，并且支持高速互联（通常是 NVLink 或 PCIe switch 直连），可以启用 GPUDirect P2P。
- `SYS`：表示两个 GPU 之间的通信必须通过系统（System）互连（例如跨 NUMA 节点的 PCIe 路径），通常不支持或性能较差，P2P 默认可能被禁用。

3）分析以上输出：

| GPU 对      | 连接类型 | 是否支持 P2P？ | 原因                                          |
| ----------- | -------- | -------------- | --------------------------------------------- |
| GPU0 ↔ GPU1 | NODE     | ✅ 是           | 同一 NUMA 节点（NUMA Affinity = 0），高速互联 |
| GPU2 ↔ GPU3 | NODE     | ✅ 是           | 同一 NUMA 节点（NUMA Affinity = 1），高速互联 |
| GPU0 ↔ GPU2 | SYS      | ❌ 否（或受限） | 跨 NUMA 节点（0 ↔ 1），需经过系统互连         |
| GPU0 ↔ GPU3 | SYS      | ❌ 否（或受限） | 跨 NUMA 节点                                  |
| GPU1 ↔ GPU2 | SYS      | ❌ 否（或受限） | 跨 NUMA 节点                                  |
| GPU1 ↔ GPU3 | SYS      | ❌ 否（或受限） | 跨 NUMA 节点                                  |

## **8. 写在最后**

GPUDirect P2P 是现代高性能 GPU 计算系统的基石技术之一。它通过消除主机内存瓶颈，使多 GPU 系统真正实现“协同作战”，为 AI、科学计算和实时处理提供关键性能支撑。在部署多卡系统时，务必检查硬件兼容性、BIOS 设置及驱动版本，以充分发挥其潜力。

