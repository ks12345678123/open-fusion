# 本周工作：Open-Fusion Replica 数据集配置与完整语义复现

日期：2026-06-22

## 一、本阶段目标

本阶段在已经完成 Windows 11 + Docker Desktop + RTX 4080 SUPER 环境配置、ICL `kt0` 完整序列复现的基础上，继续推进 Open-Fusion 官方复现流程中的 Replica 数据集部分。

本阶段目标包括：

- 下载并整理 Replica 数据集；
- 适配 Windows 文件系统下的数据目录结构；
- 验证 Docker 容器内能正确读取 Replica 场景；
- 跑通 Replica `office0` 的几何建图和 `vlfusion` 语义建图；
- 完成 `office0` 2000 帧完整序列语义复现；
- 记录当前磁盘占用和后续复现风险。

## 二、环境与硬件状态

当前复现环境如下：

```text
系统：Windows 11
Docker context：desktop-linux
GPU：NVIDIA GeForce RTX 4080 SUPER
显存：16 GB
Docker image：openfusion:local
项目路径：E:\OpenFusion
```

已验证内容：

- Docker Desktop 可用；
- Docker 容器内 GPU 可用；
- PyTorch CUDA 可用；
- Open3D、Detectron2、OpenCV、OpenFusion 可导入；
- SEEM 模型权重和 CLIP tokenizer 可加载；
- `--no-vis` headless 运行路径可用。

## 三、本阶段新增内容

### 1. Replica 下载与整理脚本

新增 Windows 原生脚本：

```text
tools/download_replica_windows.ps1
```

脚本完成以下工作：

- 从 NICE-SLAM 提供的官方 Replica 数据链接下载 `Replica.zip`；
- 解压到 `sample` 目录；
- 将目录规范化为 `sample/replica`；
- 对 `office0` 到 `office4`、`room0` 到 `room2` 共 8 个场景进行重组织；
- 将原始 `results/depth*.png` 移动为 `depth/<id>.png`；
- 将原始 `results/frame*.jpg` 移动为 `rgb/<id>.jpg`；
- 清理已搬空的 `results` 目录。

### 2. Windows 大小写目录问题修复

Replica 压缩包解压后的目录名是：

```text
sample/Replica
```

Open-Fusion 代码默认读取的是：

```text
sample/replica
```

在 Windows 文件系统中，`Replica` 和 `replica` 属于同一路径，直接 `Move-Item Replica replica` 会报错。脚本已改为通过临时目录完成大小写规范化：

```text
Replica -> replica_tmp_casefix -> replica
```

该修复避免了重复下载，也保证 Docker Linux 容器内的路径大小写与代码一致。

### 3. Git 忽略规则补充

本阶段生成的 Replica 输出目录约 100 MB，不适合直接提交到 GitHub。已在 `.gitignore` 中补充：

```text
replica_*/*
```

这样保留脚本和报告，同时避免提交生成的点云、mesh 和 glb 结果文件。

## 四、Replica 数据集状态

当前 Replica 数据集位于：

```text
E:\OpenFusion\sample\replica
```

整理后的场景帧数如下：

```text
office0：RGB 2000，Depth 2000
office1：RGB 2000，Depth 2000
office2：RGB 2000，Depth 2000
office3：RGB 2000，Depth 2000
office4：RGB 2000，Depth 2000
room0：RGB 2000，Depth 2000
room1：RGB 2000，Depth 2000
room2：RGB 2000，Depth 2000
```

当前磁盘占用约为：

```text
Replica.zip：11.59 GB
sample/replica：11.69 GB
replica_office0 输出：104.88 MB
icl_kt0 输出：49.77 MB
Docker images：20.29 GB
Docker build cache：6.87 GB
```

如果后续确认不再需要重新解压，可以删除 `Replica.zip`，可释放约 11.59 GB。

## 五、复现实验进展

### 1. Replica office0 几何短测

先运行 20 帧默认几何建图，验证数据路径、相机轨迹和输出流程：

```powershell
powershell -ExecutionPolicy Bypass -File E:\OpenFusion\tools\run_windows.ps1 --algo default --data replica --scene office0 --frames 20 --device cuda:0
```

结果：

```text
运行成功
20/20 frames completed
平均速度约 23 FPS
```

### 2. Replica office0 语义短测

随后运行 20 帧 `vlfusion`，验证 SEEM/CLIP 语义分支：

```powershell
powershell -ExecutionPolicy Bypass -File E:\OpenFusion\tools\run_windows.ps1 --algo vlfusion --data replica --scene office0 --frames 20 --device cuda:0 --no-vis
```

结果：

```text
运行成功
SEEM model loaded
20/20 frames completed
平均速度约 8.20 FPS
```

### 3. Replica office0 完整序列语义复现

完成 `office0` 2000 帧完整序列 `vlfusion`：

```powershell
powershell -ExecutionPolicy Bypass -File E:\OpenFusion\tools\run_windows.ps1 --algo vlfusion --data replica --scene office0 --device cuda:0 --no-vis
```

结果：

```text
运行成功
2000/2000 frames completed
总耗时约 176.6 秒
平均速度约 14.55 FPS
无 Open3D GUI 阻塞
```

生成输出目录：

```text
E:\OpenFusion\replica_office0
```

输出文件：

```text
color_pc.ply
semantic_pc.ply
color_mesh.ply
color_mesh.glb
```

输出规模：

```text
color_pc.ply：526,027 points
semantic_pc.ply：500,000 points
color_mesh.ply：525,944 vertices，1,041,521 triangles
```

输出文件大小：

```text
color_pc.ply：约 14.20 MB
semantic_pc.ply：约 13.50 MB
color_mesh.ply：约 40.36 MB
color_mesh.glb：约 41.91 MB
```

## 六、当前完成度

截至目前，Open-Fusion 的本机复现已完成以下内容：

- 论文和官方复现资源调研；
- Windows 11 + Docker Desktop 环境验证；
- RTX 4080 SUPER 显存需求验证；
- Docker GPU 运行验证；
- `openfusion:local` 镜像构建；
- SEEM 权重准备；
- HuggingFace tokenizer 兼容问题修复；
- Open3D GUI headless 运行参数 `--no-vis`；
- ICL 数据集下载和整理；
- ICL `kt0` 完整语义复现；
- Replica 数据集下载和整理；
- Replica `office0` 完整语义复现。

目前相当于已经完成官方 README 中最关键的本地运行链路：环境、模型、ICL 示例、Replica 示例都已经跑通。

## 七、仍需完成的工作

后续工作主要有三类。

第一，结果可视化检查：

- 在 Windows 侧打开 `color_mesh.glb` 或 `.ply`；
- 检查几何是否完整；
- 检查语义点云是否能按查询类别正确高亮；
- 如需交互式 Open3D/RViz 类显示，再单独配置 Windows X Server 或 WSLg 路径。

第二，扩展 Replica 场景：

- 继续跑 `office1` 到 `office4`；
- 继续跑 `room0` 到 `room2`；
- 对不同场景输出进行横向比较。

第三，ScanNet 和论文定量指标：

- 准备 ScanNet 数据；
- 研究官方仓库是否缺少完整评估脚本；
- 如需要复现论文表格中的 mAcc / f-mIoU，需要补充评估流程和标签映射。

## 八、阶段结论

本阶段已经证明，在当前 Windows 11 + Docker Desktop + RTX 4080 SUPER 环境下，Open-Fusion 不仅能完成 ICL 示例，也能完成 Replica 官方数据集的完整语义建图。当前机器 16 GB 显存满足该复现流程需求，`office0` 2000 帧完整语义复现运行稳定。

下一阶段建议优先做可视化检查和更多 Replica 场景复现，再决定是否投入 ScanNet 定量指标复现。
