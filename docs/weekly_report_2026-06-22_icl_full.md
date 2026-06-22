# 本周工作：Open-Fusion Windows Docker 环境配置与 ICL 完整序列复现

日期：2026-06-22

## 一、本周目标

本周的主要目标是将 Open-Fusion 从论文阅读和环境准备推进到可运行的复现状态，并在进入 Replica 数据集复现之前，完成 ICL 数据集 `kt0` 场景的完整语义建图验证。

当前阶段重点不是论文指标复现，而是确认以下内容：

- Windows 11 + Docker Desktop + NVIDIA GPU 的运行链路可用
- Open-Fusion 官方代码能够在本机环境中完成构建
- SEEM 模型和开放词汇语义路径能够正常加载和执行
- ICL 数据集能够完成几何重建和语义点云输出

## 二、已完成工作

### 1. Docker 与 GPU 环境

已完成 Docker Desktop Linux 容器环境配置，当前使用：

```text
Docker context: desktop-linux
GPU: NVIDIA GeForce RTX 4080 SUPER
VRAM: 16 GB
Docker image: openfusion:local
```

验证结果：

- `hello-world` 容器运行正常
- CUDA 容器内 `nvidia-smi` 能识别 RTX 4080 SUPER
- OpenFusion 容器内 PyTorch CUDA 可用
- Open3D、Detectron2、OpenCV、OpenFusion 模块均可导入

### 2. 项目与模型文件

官方 OpenFusion 仓库已配置在：

```text
E:\OpenFusion
```

已准备：

- OpenFusion 官方源码
- `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel` 基础镜像
- 本地复现镜像 `openfusion:local`
- SEEM 权重 `seem_focall_v1.pt`
- ICL 数据集 `kt0`、`kt1`、`kt2`、`kt3`

SEEM 权重路径：

```text
openfusion/zoo/xdecoder_seem/checkpoints/seem_focall_v1.pt
```

ICL 数据路径：

```text
sample/icl/living_room
```

### 3. Windows 兼容脚本

为了避免每次手动输入很长的 Docker 命令，新增了 Windows 运行脚本：

```text
tools/build_windows.ps1
tools/run_windows.ps1
tools/download_icl_windows.ps1
```

常用运行命令如下：

```powershell
powershell -ExecutionPolicy Bypass -File E:\OpenFusion\tools\run_windows.ps1 --algo vlfusion --data icl --scene kt0 --device cuda:0 --no-vis
```

### 4. 兼容性修复

本周解决了两个关键环境问题。

第一，官方 Dockerfile 使用 Ubuntu 官方源时，在本机网络环境下下载大量 `.deb` 包不稳定。因此新增：

```text
docker/Dockerfile.local
```

该 Dockerfile 保持官方依赖列表不变，只将 apt 源切换为清华镜像。

第二，SEEM 加载 `openai/clip-vit-base-patch32` tokenizer 时失败。原因是官方依赖中的旧版 `transformers==4.19.2` 与当前 HuggingFace 缓存行为存在兼容问题。已在 tokenizer 构建逻辑中增加 `huggingface_hub.snapshot_download` fallback。

相关修改文件：

```text
openfusion/zoo/xdecoder_seem/xdecoder/language/LangEncoder/build.py
```

此外，`main.py` 新增：

```text
--no-vis
```

用于跳过 Open3D GUI 窗口，适合 Windows Docker/headless 运行。

## 三、复现实验进展

### 1. 前期 smoke test

已完成以下验证：

- ICL `kt0` 50 帧默认几何重建
- SEEM 模型加载
- `vlfusion` 10 帧语义路径验证
- `vlfusion` 20 帧语义输出验证
- `vlfusion` 300 帧中等规模验证

这些测试均已通过。

### 2. ICL kt0 完整序列复现

本阶段进一步完成了 `ICL kt0` 完整序列的语义建图运行。

运行命令：

```powershell
powershell -ExecutionPolicy Bypass -File E:\OpenFusion\tools\run_windows.ps1 --algo vlfusion --data icl --scene kt0 --device cuda:0 --no-vis
```

输入数据：

```text
ICL living_room kt0
RGB frames: 1509
Depth frames: 1509
实际迭代数: 1508
```

运行结果：

```text
运行成功
平均速度约 13.25 FPS
无 Open3D GUI 阻塞
成功生成几何与语义输出
```

生成文件：

```text
icl_kt0/color_pc.ply
icl_kt0/color_mesh.ply
icl_kt0/color_mesh.glb
icl_kt0/semantic_pc.ply
```

输出规模：

```text
color_pc.ply: 248,485 points
semantic_pc.ply: 248,485 points
color_mesh.ply: 248,269 vertices, 489,229 triangles
```

输出文件大小：

```text
color_pc.ply: 6.40 MB
semantic_pc.ply: 6.40 MB
color_mesh.ply: 18.14 MB
color_mesh.glb: 18.83 MB
```

## 四、当前复现程度

截至目前，README 级别的 ICL 复现已经基本完成：

- 环境配置：完成
- Docker 镜像构建：完成
- SEEM 权重准备：完成
- ICL 数据准备：完成
- 几何重建路径：完成
- 语义 `vlfusion` 路径：完成
- ICL `kt0` 完整序列：完成

还没有完成的内容：

- Replica 数据集下载与定性结果复现
- 输出点云/mesh 的人工可视化检查
- ScanNet 数据准备与定量指标复现
- 论文 Table II 中 mAcc / f-mIoU 的完整评估流程

需要注意的是，官方仓库没有提供非常完整的 ScanNet 定量评估脚本，因此论文级指标复现后续可能需要补充额外评估代码。

## 五、遇到的问题

### 1. GitHub SSH 端口问题

默认 SSH 22 端口连接 GitHub 失败，已改用 GitHub SSH over 443，并通过本机 SSH key 完成仓库推送。

### 2. Docker 官方源下载不稳定

官方 Dockerfile 第一次构建时，Ubuntu 源下载大量依赖中断。通过 `Dockerfile.local` 切换镜像源后解决。

### 3. 官方下载脚本不适合 Windows 直接运行

官方 `tools/download.sh` 在 Windows 挂载目录下遇到 CRLF 和 `wget` 环境问题，因此新增 Windows 原生 ICL 下载脚本。

### 4. Open3D GUI 不适合当前 Docker 运行方式

官方 `main.py` 末尾会调用 Open3D 可视化窗口。当前阶段先以 headless 方式保存结果为主，通过 `--no-vis` 避免窗口阻塞。

## 六、下一阶段计划

下一阶段进入 Replica 数据集复现，目标是复现论文中的定性查询结果。

计划如下：

1. 下载 Replica 数据集。
2. 跑 `replica` 中的 office/room 场景。
3. 检查 `semantic_pc.ply` 和 `color_mesh.glb` 的可视化效果。
4. 对比官方论文中的查询类别，例如 sofa、lamp、cabinet、vase 等。
5. 如果 Replica 结果稳定，再考虑进入 ScanNet 定量复现。

