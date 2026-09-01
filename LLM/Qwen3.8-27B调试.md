# QWen3.8-27B调试

## 外部资源

---

> [!NOTE]
>
> 若本地已具备，可以跳过下载的过程。

外网源：http://wb.tecorigin.com:8082/repository/teco-yum-repo/release/loongnix-server/23.1/loongarch64/3.2.1/tecodriver-3.2.1-1.lns23.loongarch64.rpm

MD5码：0ceafa616548786f9bf8cfa0c7efbb54

外网源：http://wb.tecorigin.com:8082/repository/teco-yum-repo/release/loongnix-server/23.1/loongarch64/3.2.1/tecotoolkit-3.2.1-1.lns23.loongarch64.rpm

MD5码：7e220e94c7fab288a0747726705bbc5c

外网源：http://wb.tecorigin.com:8082/repository/teco-docker-tar-repo/release/loongnix-server/23.1/loongarch64/3.2.1/vllm-3.2.1-tecovllm3.2.1.tar

MD5码：814aa8b30a5c788c4ab004f547d70294

Qwen3.6-27B：https://www.modelscope.cn/models/Qwen/Qwen3.6-27B


## 模型下载
---

```bash
pip isntall modelscope
modelscope download --model Qwen/Qwen3.8-27B --local_dir ./Qwen3.8-27B
```



> [!note]
>
> 确保本机Python源设置正确



## 太初SDK升级流程

----

首先卸载旧版本产品，具体参见下面的命令。推荐先卸载内核模块，避免不必要的问题。

````bash
rmmmod aicard
rpm -qa | grep teco | xargs -i rpm -e {} --nodeps 
rpm -qa | grep sdaa | xargs -i rpm -e {} --nodeps 
````

> [!NOTE]
>
> 删除完记得检查和重启

### 常见错误

<img width="1189" height="139" alt="image-20260831175818305" src="https://github.com/user-attachments/assets/adda2a0f-4fd1-4fb2-a39b-e521385f56f0" />

### 解决方案

```bash
touch /etc/rc.d/rc.local
```

### 升级过程

```bash
[root@localhost teco]# rpm -vih tecodriver-3.2.1-1.lns23.loongarch64.rpm 
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecodriver-3.2.1-1.lns23         ################################# [100%]
Start Install Tecorigin Driver Packages.
Verifying...                          ################################# [100%]
准备中...                          [root@localhost teco]# ################################# [100%]
......
正在升级/安装...
   1:sdaadriver-3.2.0-1.lns23         ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tcml-1.15.0-1.lns23              ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecosmi-1.15.0-1.lns23           ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:TecoExporter-1.5.1-1.lns23       ################################# [100%]
End Install Tecorigin Driver Packages. Press Any Key to exit.

[root@localhost teco]# rpm -vih tecotoolkit-3.2.1-1.lns23.loongarch64.rpm
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecotoolkit-3.2.1-1.lns23        ################################# [100%]
Start Install Tecorigin Toolkit Packages.
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
[root@localhost teco]# 正在升级/安装...
   1:sdaart-3.2.0-1.lns23             ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:sdpti-1.7.0-1.lns23              ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecodevtools-3.2.0-1.lns23       ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecodnn-3.2.1-1.lns23            ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecoblas-3.2.1-1.lns23           ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecorand-3.1.0-1.lns23           ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecocustom-3.2.1-1.lns23         ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecocustom-ext-2.0.1-1.lns8      ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tccl-3.2.0-1.lns23               ################################# [ 50%]
   2:tccltest-1.1.0a1-1.lns23         ################################# [100%]
tccltest-1.1.0a1-1.lns23.loongarch64
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tccltest-1.1.0a1-1.lns23         ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tcvs-1.6.0-1.lns23               ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tsight-cli-1.12.0-1.lns23        ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tcpx-0.3.0-1.lns23               ################################# [100%]
Verifying...                          ################################# [100%]
准备中...                          ################################# [100%]
正在升级/安装...
   1:tecolmk-3.2.1-1.lns23            ################################# [100%]
End Install Tecorigin Toolkit Packages. Press Any Key to exit.

[root@localhost teco]# source /opt/tecoai/setvars.sh 
[root@localhost teco]# teco-smi 
Mon Aug 31 18:04:46 2026
+-----------------------------------------------------------------------------+
|  TECO-SMI: 1.15.0        SDAADriver: 3.2.0        SDAARuntime: 3.2.0        |
|-------------------------------+----------------------+----------------------|
| Index  Name                   | Bus-Id               | Health      SPE-Util |
|        Temp          Pwr Usage|          Memory-Usage|                      |
|=============================================================================|
|   0    TECO_AICARD_01         | 00000000:0C:00.0     | OK                0% |
|        40C                90W |        0MB / 65536MB |                      |
|-------------------------------+----------------------+----------------------|
|   1    TECO_AICARD_01         | 00000000:10:00.0     | OK                0% |
|        38C                88W |        0MB / 65536MB |                      |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  Device       PID      Process name                            Memory Usage |
|=============================================================================|
| No Process Running                                                          |
+-----------------------------------------------------------------------------+
```

> [!TIP]
>
> 安装过程不能出现任何警告和错误



## 容器创建与调试

---

### 导入镜像

```bash
docker load < vllm-3.2.1-tecovllm3.2.1.tar
```

### 官方推荐

```bash
docker run -itd --name="tecovllm_docker" --net=host --device=/dev/tcaicard0 --device=/dev/tcaicard1 --device=/dev/tcaicard2 --device=/dev/tcaicard3 --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 64g jfrog.tecorigin.net/tecotp-docker/release/loongnixserver23.1/loongarch64/vllm:3.2.1-tecovllm3.2.1 /bin/bash
```

> [!NOTE]
>
> 关于net参数，采用docker_default或host都可以
>
> device参数的个数，与/dev/tcaicardX索引相关
>
> 太初T100和I100 共用1套SDK

### 历史经验

```bash
docker run --name=Qwen3_vllm --hostname=ed957cdcc979 --volume /home/aipc/teco/checkpoint:/tecogpfs/models --network=docker_default --privileged --workdir=/softwares -p 36699:12345 --device /dev/tcaicard0:/dev/tcaicard0 --runtime=runc --detach=true -t jfrog.tecorigin.net/tecotp-docker/customer/fuyangshifan/loongnix_server23.1:2.3.0-temp tail -f /dev/null
```

| 参数                                                   | 含义                                                         | 是否必须                |
| ------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- |
| `--name=Qwen3_vllm`                                    | 为容器指定一个易于记忆的名字，这里是 `Qwen3_vllm`。          | 否 (Docker 会自动生成)  |
| `--hostname=ed957cdcc979`                              | 设置容器内部的主机名。                                       | 否                      |
| `--volume /home/aipc/teco/checkpoint:/tecogpfs/models` | 挂载数据卷，将宿主机的目录映射到容器内，实现文件共享。       | 否 (除非应用需要)       |
| `--network=docker_default`                             | 将容器连接到指定的 Docker 网络。                             | 否 (会使用默认网络)     |
| `--privileged`                                         | 赋予容器“特权模式”，拥有访问宿主机所有设备的权限，权限极高。 | 否 (仅在特殊需求时使用) |
| `--workdir=/softwares`                                 | 设置容器启动后的默认工作目录。                               | 否                      |
| `-p 36699:12345`                                       | 端口映射，将宿主机的 `36699` 端口映射到容器的 `12345` 端口。 | 否 (除非需要外部访问)   |
| `--device /dev/tcaicard0:/dev/tcaicard0`               | 将宿主机的特定设备（如此处的AI加速卡）直通给容器使用。       | 否 (除非应用需要硬件)   |
| `--runtime=runc`                                       | 指定容器运行时，`runc` 是 Docker 的默认运行时。              | 否                      |
| `--detach=true`                                        | 让容器在后台运行（Detached 模式）。                          | 否 (默认在前台运行)     |
| `-t`                                                   | 为容器分配一个伪终端 (pseudo-TTY)。                          | 否                      |
| `jfrog.tec...:2.3.0-temp`                              | 要运行的镜像名称，这是命令的核心。                           | 是                      |
| `tail -f /dev/null`                                    | 容器启动后要执行的命令，这里用于让容器保持运行状态。         | 否 (镜像有默认命令)     |

### 实际操作（推荐）

```bash
docker run -itd --name="Qwen3.8-27B" --volume /home/aipc/teco/checkpoint:/tecogpfs/models \
	--net=docker_default --workdir=/softwares -p 36670:12345 \
	--device=/dev/tcaicard0 \
	--device=/dev/tcaicard1 \
	--device=/dev/tcaicard2 \
	--device=/dev/tcaicard3 \
	--device=/dev/tcaicard4 \
	--device=/dev/tcaicard5 \
	--device=/dev/tcaicard6 \
	--device=/dev/tcaicard7 \
	--cap-add SYS_PTRACE --cap-add SYS_ADMIN \
	--shm-size 64g \
	jfrog.tecorigin.net/tecotp-docker/release/loongnixserver23.1/loongarch64/vllm:3.2.1-tecovllm3.2.1 /bin/bash
```



## 容器运行

```bash
source /opt/tecoai/setvars.sh
export HF_ENDPOINT="https://hf-mirror.com"
export VLLM_CACHE_ROOT="/work/vllm_cache"
export VLLM_TORCH_PROFILER_DIR="/work/vllm_prof"
export SDAA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


#numactl --cpunodebind=0 --membind=0 

vllm serve /tecogpfs/models/Qwen3.8-27B \
    --served-model-name Qwen3.8-27B \
    --trust-remote-code \
    --tensor_parallel_size 8 \
    --port 8888 \
    --gpu-memory-utilization 0.9 \
    --max_num_seqs 32 \
    --max_model_len 262144 \
    --quantization sdaa_wint8 \
    --kv-cache-dtype fp8_e5m2 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max_num_batched_tokens 8192 \
    --mamba-cache-mode align \
    --compilation_config='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

