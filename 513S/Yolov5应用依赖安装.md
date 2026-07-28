## 将513S Demo 迁移到La平台

---



## numpy 2.2.6 安装

---

### 基本背景

---

从仓库中拉取的压缩包，解压后效果如下：

```bash
/home/loongson/2K3000-AI/requires/numpy-2.2.6/numpy

loongson@loongson-pc:~/2K3000-AI/requires/numpy-2.2.6/numpy$ ls
_array_api_info.py   __config__.py.in  ctypeslib.py           dtypes.py               f2py                    __init__.py   matlib.pyi    _pytesttester.py   testing     version.pyi
_array_api_info.pyi  _configtool.py    ctypeslib.pyi          dtypes.pyi              fft                     __init__.pyi  matrixlib     _pytesttester.pyi  tests
_build_utils         _configtool.pyi   _distributor_init.py   exceptions.py           _globals.py             lib           meson.build   py.typed           _typing
char                 conftest.py       _distributor_init.pyi  exceptions.pyi          _globals.pyi            linalg        polynomial    random             typing
compat               _core             distutils              _expired_attrs_2_0.py   __init__.cython-30.pxd  ma            __pycache__   rec                _utils
__config__.pyi       core              doc                    _expired_attrs_2_0.pyi  __init__.pxd            matlib.py     _pyinstaller  strings            version.py
loongson@loongson-pc:~/2K3000-AI/requires/numpy-2.2.6/numpy$

```

### numpy 2.2.6源码构建

```bash
cd numpy-2.2.6

pip install . -v # 自动解决依赖，前提正确配置Python仓库

## 验证
python -c "import numpy; print(numpy.get_include())"
```



### Numpy API 兼容问题

```bash
(513s_env) loongson@loongson-pc:~/2K3000-AI/513s$ sh -x RUN.sh
+ python main.py --source ./test30.mp4 --model yolov5n_quant.onnx --show --save output-gpu2.mp4 --device cpu
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
Traceback (most recent call last):  File "/home/loongson/2K3000-AI/513s/main.py", line 28, in <module>
    from detector import YOLOv5ONNXDetector
  File "/home/loongson/2K3000-AI/513s/detector.py", line 17, in <module>
    import onnxruntime as ort
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/onnxruntime/__init__.py", line 24, in <module>
    from onnxruntime.capi._pybind_state import (
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/onnxruntime/capi/_pybind_state.py", line 32, in <module>
    from .onnxruntime_pybind11_state import *  # noqa
Traceback (most recent call last):
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy/core/_multiarray_umath.py", line 44, in __getattr__
    raise ImportError(msg)
ImportError:
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
Traceback (most recent call last):
  File "/home/loongson/2K3000-AI/513s/main.py", line 28, in <module>
    from detector import YOLOv5ONNXDetector
  File "/home/loongson/2K3000-AI/513s/detector.py", line 17, in <module>
    import onnxruntime as ort
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/onnxruntime/__init__.py", line 61, in <module>
    raise import_capi_exception
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/onnxruntime/__init__.py", line 24, in <module>
    from onnxruntime.capi._pybind_state import (
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/onnxruntime/capi/_pybind_state.py", line 32, in <module>
    from .onnxruntime_pybind11_state import *  # noqa
ImportError
```



#### 简单分析

```bash
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

`onnxruntime` 这个二进制包是针对 NumPy 1.x 的C-API编译的，但环境里装的是 **NumPy 2.2.6**，NumPy 2.0 对C-API做了不兼容的改动，导致所有没针对NumPy2重新编译的C扩展（这里是onnxruntime的`onnxruntime_pybind11_state`模块）一加载就崩。



#### 解决方案

```bash
pip uninstall -y numpy
pip install "numpy==1.26.4" ## 特定版本的确定，有模型根据当前513s_env已安装组件确认

### 确认
(513s_env) loongson@loongson-pc:~/2K3000-AI$ pip3 list
Package            Version
------------------ -----------------------
coloredlogs        15.0.1
Cython             3.2.9
flatbuffers        20230107074616
humanfriendly      10.0
meson              1.11.2
meson-python       0.20.0
mpmath             1.3.0
numpy              1.26.4
onnx               1.17.0
onnxruntime-lacm   1.21.0+lacm.0.1.0.lnd.4
packaging          26.2
pip                26.1.2
protobuf           7.35.1
pyproject-metadata 0.12.1
setuptools         79.0.1
sympy              1.14.0
tomli              2.4.1
(513s_env) loongson@loongson-pc:~/2K3000-AI$ python -c "import numpy; print(numpy.__version__)"
1.26.4
(513s_env) loongson@loongson-pc:~/2K3000-AI$ python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
1.21.0+lacm.0.1.0.lnd.4
['CUDAExecutionProvider', 'CPUExecutionProvider']
```



### _multiarray_umath 问题



#### 报错现象

```bash
Traceback (most recent call last):
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy/core/__init__.py", line 24, in <module>
    from . import multiarray
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy/core/multiarray.py", line 10, in <module>
    from . import overrides
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy/core/overrides.py", line 8, in <module>
    from numpy.core._multiarray_umath import (
ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy/__init__.py", line 130, in <module>
    from numpy.__config__ import show as show_config
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy/__config__.py", line 4, in <module>
    from numpy.core._multiarray_umath import (
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy/core/__init__.py", line 50, in <module>
    raise ImportError(msg)
.... .... 
```



#### 简单分析

```bash
ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
```

初步判断numpy 1.26.4 装完之后，编译出来的C扩展模块（`_multiarray_umath`）根本不存在



#### 排查过程

1. 确认C扩展是否构建出来

```bash
(513s_env) loongson@loongson-pc:~/2K3000-AI$ python3 -c "
> import numpy, os, glob
> core_dir = os.path.join(os.path.dirname(numpy.__file__), 'core')
> print('numpy路径:', os.path.dirname(numpy.__file__))
> print('找到的扩展文件:', glob.glob(os.path.join(core_dir, '_multiarray_umath*')))
> "
numpy路径: /home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy
找到的扩展文件: ['/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy/core/_multiarray_umath.cpython-310-loongarch64-linux-gnu.so', '/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/numpy/core/_multiarray_umath.cpython-310.so']
```



```bash
(513s_env) loongson@loongson-pc:~/2K3000-AI/513s$ pip show -f numpy | grep -i -E "multiarray|\.so"
  numpy.libs/libgfortran-9f255ad8.so.5.0.0
  numpy.libs/libopenblas-r0-3f394acb.3.15.so
  numpy/_core/__pycache__/_multiarray_umath.cpython-310.pyc
  numpy/_core/__pycache__/multiarray.cpython-310.pyc
  numpy/_core/_multiarray_umath.py
  numpy/_core/multiarray.py
  numpy/core/__pycache__/multiarray.cpython-310.pyc
  numpy/core/_multiarray_tests.cpython-310-loongarch64-linux-gnu.so
  numpy/core/_multiarray_umath.cpython-310-loongarch64-linux-gnu.so
  numpy/core/_operand_flag_tests.cpython-310-loongarch64-linux-gnu.so
  numpy/core/_rational_tests.cpython-310-loongarch64-linux-gnu.so
  numpy/core/_simd.cpython-310-loongarch64-linux-gnu.so
  numpy/core/_struct_ufunc_tests.cpython-310-loongarch64-linux-gnu.so
  numpy/core/_umath_tests.cpython-310-loongarch64-linux-gnu.so
  numpy/core/include/numpy/__multiarray_api.c
  numpy/core/include/numpy/__multiarray_api.h
  numpy/core/multiarray.py
  numpy/core/multiarray.pyi
  numpy/core/tests/__pycache__/test_multiarray.cpython-310.pyc
  numpy/core/tests/test_multiarray.py
  numpy/fft/_pocketfft_internal.cpython-310-loongarch64-linux-gnu.so
  numpy/linalg/_umath_linalg.cpython-310-loongarch64-linux-gnu.so
  numpy/linalg/lapack_lite.cpython-310-loongarch64-linux-gnu.so
  numpy/matrixlib/tests/__pycache__/test_multiarray.cpython-310.pyc
  numpy/matrixlib/tests/test_multiarray.py
  numpy/random/_bounded_integers.cpython-310-loongarch64-linux-gnu.so
  numpy/random/_common.cpython-310-loongarch64-linux-gnu.so
  numpy/random/_generator.cpython-310-loongarch64-linux-gnu.so
  numpy/random/_mt19937.cpython-310-loongarch64-linux-gnu.so
  numpy/random/_pcg64.cpython-310-loongarch64-linux-gnu.so
  numpy/random/_philox.cpython-310-loongarch64-linux-gnu.so
  numpy/random/_sfc64.cpython-310-loongarch64-linux-gnu.so
  numpy/random/bit_generator.cpython-310-loongarch64-linux-gnu.so
  numpy/random/mtrand.cpython-310-loongarch64-linux-gnu.so
  numpy/typing/tests/data/fail/multiarray.pyi
  numpy/typing/tests/data/pass/__pycache__/multiarray.cpython-310.pyc
  numpy/typing/tests/data/pass/multiarray.py
  numpy/typing/tests/data/reveal/multiarray.pyi
(513s_env) loongson@loongson-pc:~/2K3000-AI/513s$
```

`pip show -f` 显示 **该文件确实存在**：

```bash
numpy/core/_multiarray_umath.cpython-310-loongarch64-linux-gnu.so
```

架构标记（`loongarch64-linux-gnu`）也是对的，说明龙芯官方源提供的这个1.26.4 wheel本身包装是正确的、针对性编译的.



2. 确认Python解释器扩展模块名后缀标签(ABI tag)

```bash
(513s_env) loongson@loongson-pc:~/2K3000-AI$ python3 -c "
> import sysconfig, importlib.machinery
> print('EXT_SUFFIX:', sysconfig.get_config_var('EXT_SUFFIX'))
> print('SOABI:', sysconfig.get_config_var('SOABI'))
> print('识别的扩展后缀:', importlib.machinery.EXTENSION_SUFFIXES)
> "
EXT_SUFFIX: .cpython-310.so
SOABI: cpython-310
识别的扩展后缀: ['.cpython-310.so', '.abi3.so', '.so']
```



3. 根因分析

```bash
EXT_SUFFIX: .cpython-310.so

这不对——正常的Debian/Ubuntu系Linux上，Python C扩展的后缀应该带完整的架构三元组，形如 .cpython-310-loongarch64-linux-gnu.so（这也正是磁盘上那个numpy .so 文件实际使用的命名）。但你这个venv里的Python解释器，EXT_SUFFIX 却只有精简的 .cpython-310.so，少了架构标签这一段。

Python的导入机制不是"扫描目录里所有.so文件模糊匹配"，而是精确拼出候选文件名去找：对于 _multiarray_umath 这个模块，当前这个解释器会依次找：

_multiarray_umath.cpython-310.so
_multiarray_umath.abi3.so
_multiarray_umath.so

而磁盘上实际的文件名是 _multiarray_umath.cpython-310-loongarch64-linux-gnu.so——上面三个候选一个都对不上，所以哪怕文件明明就在旁边、内容完全合法，Python还是会报"找不到模块"。这就是你看到的诡异现象的真正原因：这个venv所用的python3.10解释器，跟编译这个numpy wheel时所针对的python3.10，不是同一套ABI配置
```



```bash
(513s_env) loongson@loongson-pc:~/2K3000-AI$ gcc -print-multiarch
loongarch64-linux-gnu

```

#### 解决方案

**Python的模块名后缀（如 `.cpython-310-loongarch64-linux-gnu.so`）只是"导入机制用来在目录里认哪些文件是候选扩展模块"的命名约定，跟`.so`文件内部的二进制内容/符号完全无关。**

 真正决定这个`.so`能不能被正确加载的，是它内部导出的 `PyInit_<模块名>` 这个符号（跟文件名字无关），dlopen加载库、找这个symbol，跟文件叫什么名字是两回事。

所以只要建一个"别名"文件名让Python的导入机制认出来它，加载起来是完全正常的，不存在"表面上骗过了但运行时会崩"的风险。



因为不止numpy一个包受影响——onnxruntime自己的`onnxruntime_pybind11_state`、opencv的`cv2`模块，只要是编译扩展，都会撞上同样的问题，一个个手动建软链接太麻烦：

````bash
#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# fix_ext_suffix.sh
#
# 背景：在部分源码编译的 Python(尤其编译时gcc不支持 -print-multiarch 的情况，
# 常见于龙芯LoongArch等非Debian官方gcc工具链环境)下，Python解释器探测到的
# EXT_SUFFIX 会缺少完整的平台三元组，变成精简版 ".cpython-310.so"，
# 而PyPI/龙芯官方源分发的numpy/onnxruntime/opencv等预编译wheel里的.so文件，
# 命名都是完整版 ".cpython-310-loongarch64-linux-gnu.so"，两者对不上，
# 导致 Python 明明能看到.so文件、文件也是合法的，却报 ModuleNotFoundError。
#
# 本脚本给每个"完整三元组后缀"的.so文件，在同目录下建一个"精简后缀"的软链接，
# 软链接不改变文件内容(dlopen真正依赖的是.so内部的PyInit_<module>符号，
# 跟文件名字符串无关)，只是让Python的导入机制能找到它。
#
# 用法：
#   source 你的venv后，每次新装/升级了带C扩展的包(numpy/onnxruntime/opencv-python/
#   scipy/pandas等)，都重新跑一次本脚本：
#     bash fix_ext_suffix.sh
#
# 注意：这是绕开症状的兼容方案，不是根治办法。长期使用建议排查
# 源码编译Python时使用的gcc是否支持 `-print-multiarch`，
# 或改用apt/系统包管理器提供的、ABI配置完整的Python重建虚拟环境。

set -euo pipefail

# 目标"精简"后缀：优先取当前解释器实际探测到的EXT_SUFFIX
TARGET_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
if [ -z "$TARGET_SUFFIX" ]; then
    echo "错误：无法从当前python3获取 EXT_SUFFIX，请确认已激活正确的虚拟环境" >&2
    exit 1
fi

echo "当前解释器识别的EXT_SUFFIX: $TARGET_SUFFIX"

if [[ "$TARGET_SUFFIX" == *"-linux-gnu.so" ]]; then
    echo "当前EXT_SUFFIX已经是完整版，通常不需要本脚本，直接退出。"
    exit 0
fi

VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "site-packages路径: $VENV_SITE"

count=0
# 匹配site-packages下所有"完整三元组后缀"的.so文件(不写死loongarch64，
# 兼容其他架构下遇到同类问题，例如 x86_64-linux-gnu / aarch64-linux-gnu)
while IFS= read -r -d '' f; do
    dir=$(dirname "$f")
    base=$(basename "$f")
    # 提取模块名前缀(去掉形如 .cpython-310-<arch>-linux-gnu.so 的完整后缀)
    prefix=$(echo "$base" | sed -E 's/\.cpython-[0-9]+-[a-zA-Z0-9_]+-linux-gnu\.so$//')
    if [ "$prefix" == "$base" ]; then
        continue  # 文件名不匹配这个模式，跳过
    fi
    short="${prefix}${TARGET_SUFFIX}"
    if [ ! -e "$dir/$short" ]; then
        ln -s "$base" "$dir/$short"
        echo "  建立软链接: $dir/$short -> $base"
        count=$((count + 1))
    fi
done < <(find "$VENV_SITE" -name "*-linux-gnu.so" -print0)

echo "完成，本次新建了 $count 个软链接。"
echo ""
echo "建议现在验证一下关键包能否正常import，例如："
echo "  python3 -c \"import numpy; print(numpy.__version__)\""
echo "  python3 -c \"import onnxruntime as ort; print(ort.get_available_providers())\""
echo "  python3 -c \"import cv2; print(cv2.__version__)\""

````

验证

```bash
(513s_env) loongson@loongson-pc:~/2K3000-AI$ python3 -c "import numpy; print(numpy.__version__); import numpy as np; print(np.array([1,2,3])*2)"
1.26.4
[2 4 6]
```

#### 注意事项

----

**这个软链接不是一劳永逸的**——以后任何时候升级/重装 numpy、onnxruntime、opencv-python，或者新装任何带C扩展的包（比如scipy、pandas），新装进来的`.so`文件不会自动带软链接，得重新跑一遍这个脚本。

建议把它存成一个固定脚本（比如 `fix_ext_suffix.sh`），放进项目里，写进部署文档，每次装完依赖都跑一下。

**这只是绕开症状，没解决病根**——真正的病根是这个源码编译的Python的`EXT_SUFFIX`本身就配置不完整，之后只要是"编译扩展模块"这个大类的包，都会反复撞上同一个问题。如果这台设备后续还要长期维护、装更多依赖，我还是建议找机会按之前说的排查一下`gcc -print-multiarch`、看能不能用apt装到一个ABI正常的python3.10——软链接方案适合现在先把测试跑起来，不建议当成长期方案。



## onnx 安装遇到的Python ABI Tag问题

----



### 报错现象

----



```bash
(513s_env) loongson@loongson-pc:~/2K3000-AI/513s$ pip install --no-deps /tmp/onnx_check/v1_17/onnx-1.17.0-cp310-cp310-manylinux_2_28_loongarch64.whl
Looking in indexes: https://pypi.loongnix.cn/loongson/pypi/+simple, https://pypi.tuna.tsinghua.edu.cn/simple
Processing /tmp/onnx_check/v1_17/onnx-1.17.0-cp310-cp310-manylinux_2_28_loongarch64.whl
Installing collected packages: onnx
Successfully installed onnx-1.17.0
(513s_env) loongson@loongson-pc:~/2K3000-AI/513s$
(513s_env) loongson@loongson-pc:~/2K3000-AI/513s$ # 验证能否正常import(重点看有没有protobuf相关报错)
(513s_env) loongson@loongson-pc:~/2K3000-AI/513s$ python3 -c "import onnx; print('onnx版本:', onnx.__version__)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/loongson/2K3000-AI/513s_env/lib/python3.10/site-packages/onnx/__init__.py", line 77, in <module>
    from onnx.onnx_cpp2py_export import ONNX_ML
ImportError: cannot import name 'ONNX_ML' from 'onnx.onnx_cpp2py_export' (unknown location)
```



### 简单分析



这个问题也是和Python3 编译器的EXT_SUFFIX截断相关。

这次的报错其实和之前处理过的numpy/onnxruntime/opencv是**同一类问题**——刚装的 `onnx` 也带了原生编译扩展（`onnx_cpp2py_export`），大概率又是之前那个"EXT_SUFFIX截断"的坑：wheel里的`.so`文件名是完整版（`onnx_cpp2py_export.cpython-310-loongarch64-linux-gnu.so`），但你这个源码编译的Python解释器只认精简版后缀（`.cpython-310.so`），两者对不上，Python的导入机制退化成了某种"找到了名字但没有真实内容"的异常状态（"unknown location"这个措辞就是这个特征），跟当初 `cv2.__file__` 显示 `None` 是同一种现象。



### 排查过程



先确认一下文件名：

```bash
find /home/loongson/2K3000-AI/513s_env -iname "onnx_cpp2py_export*"
```

预期能看到类似 `onnx_cpp2py_export.cpython-310-loongarch64-linux-gnu.so` 这种完整后缀的文件。

如果确实如此，直接**重新跑一遍之前那个软链接脚本**就行——它是通用的，专门设计成"每次装完新的带C扩展的包都重跑一次"：

```bash
bash fix_ext_suffix.sh
```

跑完立刻验证：

```bash
python3 -c "import onnx; print('onnx版本:', onnx.__version__)"
(513s_env) loongson@loongson-pc:~/2K3000-AI$ python3 -c "import onnx; print('onnx版本:', onnx.__version__)"
onnx版本: 1.17.0
(513s_env) loongson@loongson-pc:~/2K3000-AI$
```
