##  背景介绍
---

<img width="1080" height="608" alt="640" src="https://github.com/user-attachments/assets/036af077-84e8-4200-ba0d-9bd64aec6594" />

GPUDirect P2P（Peer-to-Peer）是 NVIDIA 开发的一项关键 GPU 通信技术，旨在提升多 GPU 系统中显存间数据传输的效率。它通过绕过主机内存（Host Memory）和 CPU 的中转，实现 GPU 与 GPU 之间的直接点对点通信，从而显著降低延迟、提高带宽，并减少系统资源开销。

主要内容：

1. 技术背景
2. GPUDirect P2P工作原理
3. GPUDirect P2P关键技术特性
4. GPUDirect P2P启用条件
5. GPUDirect P2P典型应用场景
6. GPUDirect P2P限制与挑战
7. GPUDirect P2P启用验证
8. 写在最后
