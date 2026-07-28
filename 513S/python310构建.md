# Python 3.10.19源码构建指南

---



## 背景

----

验证Yolov5算法，需要高版本Python；且原来的Python缺少OpenSSL支持

```bash
使用pip3 安装whl包的报错信息：
ould not fetch URL https://pypi.loongnix.cn/loongson/pypi/+simple/numpy/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.loongnix.cn', port=443): Max retries exceeded with url: /loongson/pypi/+simple/numpy/ (Caused by SSLError("Can't connect to HTTPS URL because the SSL module is not available.")) - skipping
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/numpy/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/numpy/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/numpy/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/numpy/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/numpy/
Could not fetch URL https://pypi.tuna.tsinghua.edu.cn/simple/numpy/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.tuna.tsinghua.edu.cn', port=443): Max retries exceeded with url: /simple/numpy/ (Caused by SSLError("Can't connect to HTTPS URL because the SSL module is not available.")) - skipping
ERROR: Could not find a version that satisfies the requirement numpy (from opencv) (from versions: none)
ERROR: No matching distribution found for numpy

```



## 解决方案

----

### 一、准备工作：安装编译依赖

不同发行版命令不同，以 **Ubuntu/Debian** 为例：

```bash
sudo apt update
sudo apt install -y build-essential gdb lcov pkg-config \
  libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
  libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
  lzma lzma-dev tk-dev uuid-dev zlib1g-dev libnss3-dev \
  wget curl xz-utils
```

**CentOS/RHEL/Rocky**：

```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y openssl-devel bzip2-devel libffi-devel \
  zlib-devel xz-devel sqlite-devel ncurses-devel readline-devel \
  gdbm-devel tk-devel uuid-devel wget
```

> **关键点**：`libssl-dev`（或 `openssl-devel`）必须提前装好，否则 `configure` 阶段检测不到 OpenSSL，编译出的 Python 会缺少 `ssl` 模块（`pip`、`https` 请求都会失败）。

### 二、下载并解压源码

```bash
cd /usr/local/src
wget https://www.python.org/ftp/python/3.10.19/Python-3.10.19.tgz
tar -xzf Python-3.10.19.tgz
cd Python-3.10.19
```

### 三、configure 阶段（重点）

推荐配置命令：

```bash
./configure \
  --prefix=/usr/local/python3.10.19 \
  --enable-optimizations \
  --with-lto \
  --enable-shared \
  --with-ensurepip=install \
  --with-openssl=/usr \
  --with-system-ffi \
  --with-computed-gotos \
  LDFLAGS="-Wl,-rpath=/usr/local/python3.10.19/lib"
```

逐项说明：

| 参数                       | 作用                                                         | 是否建议开启                               |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| `--prefix`                 | 指定安装路径，避免覆盖系统 Python                            | **必须**，建议独立目录                     |
| `--enable-optimizations`   | 开启 PGO（Profile Guided Optimization），编译后运行基准测试再重新优化编译，性能提升约 10%~20% | 推荐，但会显著增加编译时间（可能翻倍以上） |
| `--with-lto`               | 开启链接时优化（Link Time Optimization），进一步提升性能     | 推荐，与 `--enable-optimizations` 搭配使用 |
| `--enable-shared`          | 生成 `libpython3.10.so` 共享库，供其他程序（如嵌入式调用、mod_wsgi）链接 | 视需求，一般生产环境建议开启               |
| `--with-openssl=/path`     | **显式指定 OpenSSL 路径**，若系统 OpenSSL 版本较新（1.1.1+/3.x）或安装在非标准路径，务必指定 | **重要**，见下方详细说明                   |
| `--with-ensurepip=install` | 编译完成自带安装 pip                                         | 推荐                                       |
| `--with-system-ffi`        | 使用系统 libffi（ctypes 模块依赖）                           | 推荐                                       |
| `--with-computed-gotos`    | 解释器主循环使用 computed goto，提升字节码执行速度           | 推荐（默认在支持的编译器下已开启）         |
| `LDFLAGS -Wl,-rpath`       | 若开启 `--enable-shared`，运行时能找到 so 库，避免 `error while loading shared libraries` | 开启 shared 时必须加                       |

#### 关于 OpenSSL 的特别说明

Python 3.10 要求 **OpenSSL 1.1.1 及以上**版本（推荐 1.1.1 或 3.0.x）。有两种情况：

**情况1：使用系统自带的 OpenSSL（版本满足要求）**

```bash
./configure --with-openssl=/usr ...
```

**情况2：系统 OpenSSL 版本过低或需要自行编译新版 OpenSSL**

先单独编译安装 OpenSSL 到自定义目录，例如 `/usr/local/openssl-3.0`，然后：

```bash
./configure \
  --with-openssl=/usr/local/openssl-3.0 \
  --with-openssl-rpath=auto \
  ...
```

`--with-openssl-rpath=auto` 会自动把 OpenSSL 的 lib 路径写入 rpath，避免运行时找不到 `libssl.so`/`libcrypto.so`。

#### 配置完成后检查

`configure` 结束后务必查看输出摘要，确认：

```
checking for --enable-optimizations... yes
checking for openssl/ssl.h... yes（不是 no）
```

若看到类似：

```
The necessary bits to build these optional modules were not found:
_ssl
```

说明 OpenSSL 没检测到，需要检查 `--with-openssl` 路径或依赖是否装全，重新执行 configure。

### 四、编译与安装

```bash
# 使用多核加速编译
nproc  # 查看 CPU 核心数
make -j$(nproc)
```

> 若开启了 `--enable-optimizations`，`make` 过程中会自动跑一遍 profile task（类似跑一遍 pyperformance 基准测试集），耗时会明显增加（普通机器可能 30 分钟以上），属于正常现象。

安装（**强烈建议 altinstall，避免覆盖系统自带 python3**）：

```bash
sudo make altinstall
```

- `make install` 会创建/覆盖 `python3`、`pip3` 等软链接，容易破坏系统依赖 Python 的工具（如 yum/dnf）。
- `make altinstall` 只会安装为 `python3.10`、`pip3.10`，不影响系统默认 python。

### 五、验证安装

```bash
/usr/local/python3.10.19/bin/python3.10 --version
/usr/local/python3.10.19/bin/python3.10 -m ssl  # 验证ssl模块可用
/usr/local/python3.10.19/bin/python3.10 -c "import ssl; print(ssl.OPENSSL_VERSION)"
```

正常应输出类似：

```bash
Python 3.10.19
OpenSSL 3.0.x ...
loongson@loongson-pc:~$ /usr/local/python3.10.19/bin/python3.10 -c "import ssl; print(ssl.OPENSSL_VERSION)"
OpenSSL 1.1.1d  10 Sep 2019
```

若提示 `ModuleNotFoundError: No module named '_ssl'`，说明编译时 OpenSSL 未正确链接，需要回到 configure 阶段重新排查。

### 六、可选：设置环境变量 / 软链接

```bash
sudo ln -s /usr/local/python3.10.19/bin/python3.10 /usr/local/bin/python3.10
sudo ln -s /usr/local/python3.10.19/bin/pip3.10 /usr/local/bin/pip3.10
```

或加入 `PATH`：

```bash
echo 'export PATH=/usr/local/python3.10.19/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

echo '/usr/local/python3.10.19/lib' | sudo tee /etc/ld.so.conf.d/python3.10.conf
sudo ldconfig
```

### 七、常见坑总结

1. **忘记装 `libssl-dev`**：导致 `_ssl` 模块编译不出来，pip 无法使用 https 源。
2. **`--enable-shared` 但没配 rpath**：运行时报 `error while loading shared libraries: libpython3.10.so.1.0`。
3. **`--enable-optimizations` 但机器性能弱/时间紧**：可以先不开，日常开发用普通编译，生产环境部署时再开优化重新编译。
4. **直接 `make install` 覆盖系统 Python**：可能导致系统工具（如 yum、apt 相关脚本）报错，务必用 `altinstall`。
5. **多个 Python 版本共存管理混乱**：建议用不同 `--prefix` 隔离，或后续引入 `pyenv` 统一管理。



## 后记 环境变量设置

---



```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
timeout = 60
index-url = https://pypi.loongnix.cn/loongson/pypi/+simple
extra-index-url = https://pypi.tuna.tsinghua.edu.cn/simple

[install]
trusted-host =
        pypi.loongnix.cn
        pypi.tuna.tsinghua.edu.cn
EOF

### .bashrc
export PATH=/usr/local/python3.10.19/bin:$PATH
export PKG_CONFIG_PATH=/opt/opencv410/lib/pkgconfig:$PKG_CONFIG_PATH
export PYTHONPATH=/opt/opencv410/lib/python3.10/site-packages/cv2/python-3.10:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/opencv410/lib:/usr/lib/loongarch64-linux-gnu:$LD_LIBRARY_PATH

loongson@loongson-pc:~$ cat /etc/ld.so.conf.d/python3.10.conf
/usr/local/python3.10.19/lib

```
