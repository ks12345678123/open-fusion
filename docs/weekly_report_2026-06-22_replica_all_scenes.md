# 本周工作：Open-Fusion Replica 全场景语义复现完成

日期：2026-06-22

## 一、本阶段目标

本阶段继续推进 Open-Fusion 官方复现流程，在已经完成 ICL `kt0` 和 Replica `office0` 的基础上，将 Replica 数据集中的全部官方场景都跑完整语义建图。

目标包括：

- 删除不再需要的 Replica 压缩包以释放磁盘空间；
- 完成 Replica 剩余 7 个场景的 `vlfusion` 复现；
- 汇总 8 个 Replica 场景的输出情况；
- 确认当前本机复现已经覆盖官方 README 中的主要定性数据流程。

## 二、存储空间处理

Replica 数据集此前保留了两份内容：

```text
sample/Replica.zip：约 11.59 GB
sample/replica：约 11.69 GB
```

经检查，`sample/replica` 中 8 个场景均已完整解压和整理：

```text
office0~office4：每个场景 RGB 2000 / Depth 2000 / traj.txt 存在
room0~room2：每个场景 RGB 2000 / Depth 2000 / traj.txt 存在
```

因此已安全删除：

```text
E:\OpenFusion\sample\Replica.zip
```

删除后，OpenFusion 工作目录占用从约 27.32 GB 降低到约 16.61 GB。

## 三、Replica 全场景复现命令

每个场景均使用相同的 Docker 运行入口：

```powershell
powershell -ExecutionPolicy Bypass -File E:\OpenFusion\tools\run_windows.ps1 --algo vlfusion --data replica --scene <scene> --device cuda:0 --no-vis
```

其中 `<scene>` 覆盖以下 8 个场景：

```text
office0
office1
office2
office3
office4
room0
room1
room2
```

所有运行均为 2000 帧完整序列，没有使用帧数截断参数。

## 四、运行结果

Replica 全部官方场景均已完成完整 `vlfusion` 语义复现：

```text
office0：2000/2000 frames completed
office1：2000/2000 frames completed
office2：2000/2000 frames completed
office3：2000/2000 frames completed
office4：2000/2000 frames completed
room0：2000/2000 frames completed
room1：2000/2000 frames completed
room2：2000/2000 frames completed
```

运行过程中出现过 Open3D 写 PLY 的颜色 clamp warning：

```text
Open3D WARNING: Write Ply clamped color value to valid range
```

该 warning 出现在输出阶段，不影响程序完成，也没有导致结果文件缺失。

## 五、输出文件检查

每个场景均生成了以下 4 个输出文件：

```text
color_pc.ply
semantic_pc.ply
color_mesh.ply
color_mesh.glb
```

输出目录和大小如下：

```text
replica_office0：104.88 MB
replica_office1： 61.19 MB
replica_office2：128.08 MB
replica_office3：176.31 MB
replica_office4：155.43 MB
replica_room0  ：166.02 MB
replica_room1  ：101.40 MB
replica_room2  ：112.60 MB
```

Replica 全场景输出合计约 1.0 GB。

## 六、当前整体复现进度

截至目前，本机复现已经完成：

- Windows 11 + Docker Desktop 环境配置；
- Docker GPU 可用性验证；
- RTX 4080 SUPER 16 GB 显存需求验证；
- `openfusion:local` 镜像构建；
- SEEM 模型权重准备；
- HuggingFace tokenizer 兼容问题修复；
- Open3D headless 运行参数 `--no-vis`；
- ICL 数据集下载与整理；
- ICL `kt0` 完整语义复现；
- Replica 数据集下载与整理；
- Replica 8 个官方场景完整语义复现。

这说明当前机器已经能稳定运行 Open-Fusion 官方仓库中的主要定性复现流程。

## 七、当前磁盘占用

当前主要占用如下：

```text
E:\OpenFusion 总占用：约 16.61 GB
sample/replica：约 11.69 GB
sample/icl：约 2.31 GB
Replica 全场景输出：约 1.0 GB
Docker images：约 20.29 GB
Docker build cache：约 6.87 GB
```

复现相关总占用约为：

```text
16.61 GB + 20.29 GB + 6.87 GB = 43.77 GB
```

当前没有继续保留 `Replica.zip`，因此后续如果需要重新从压缩包解压 Replica，需要重新下载。

## 八、下一阶段计划

下一阶段有两个可选方向。

第一，做可视化和结果检查：

- 打开每个场景的 `color_mesh.glb`；
- 检查重建几何是否完整；
- 检查 `semantic_pc.ply` 的语义查询结果；
- 对照论文中展示的定性结果进行人工比对。

第二，进入 ScanNet 定量复现：

- 准备 ScanNet 数据集；
- 检查官方仓库缺失的评估脚本部分；
- 补全标签映射和指标统计流程；
- 尝试复现论文表格中的 mAcc / f-mIoU。

从风险角度看，建议先完成 Replica 输出可视化检查，再进入 ScanNet，因为 ScanNet 定量复现比当前定性流程需要更多数据准备和评估代码补充。

## 九、阶段结论

本阶段已经完成 Replica 数据集全部 8 个官方场景的完整语义复现。结合此前完成的 ICL `kt0` 完整复现，Open-Fusion 在当前 Windows 11 + Docker + RTX 4080 SUPER 环境下的主要官方示例链路已经跑通。

后续工作重点应从“能否运行”转向“结果质量检查”和“论文定量指标复现”。
