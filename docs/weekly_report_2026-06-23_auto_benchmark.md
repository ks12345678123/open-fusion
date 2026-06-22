# 本周工作：Replica 全场景自动查询基准构建与验证

## 一、本周目标

本周在已完成 Open-Fusion 复现、对象级地图原型、DeepSeek 查询 Agent 和查询结果可视化的基础上，继续推进实验评估部分。核心目标是把原先 `replica_office0` 的 6 条 seed 查询扩展为覆盖 Replica 8 个场景的查询基准初稿，并建立可重复运行的 benchmark 流程。

## 二、已完成工作

### 1. 补齐 Replica 全场景语义标签映射

此前只有 `replica_office0` 重新运行后生成了 `semantic_label_map.json`，其余场景的对象级地图仍显示为 `semantic_color_000` 这类占位标签，不适合写入论文实验。

本次重新运行了以下 7 个场景：

```text
replica_office1
replica_office2
replica_office3
replica_office4
replica_room0
replica_room1
replica_room2
```

结果：Replica 8 个场景均已生成 `semantic_label_map.json`，后续 object_map 可以使用 `chair`、`table`、`sofa`、`door`、`tv` 等真实语义标签。

### 2. 重建 8 个场景对象级地图

使用当前 `openfusion_agent.object_map` 对 8 个 Replica 场景重新生成对象级地图。

重建结果：

```text
replica_office0: 25 objects
replica_office1: 21 objects
replica_office2: 26 objects
replica_office3: 37 objects
replica_office4: 25 objects
replica_room0:   38 objects
replica_room1:   28 objects
replica_room2:   22 objects
```

标签检查结果：

```text
Fallback semantic_color_*: 0
```

这说明当前 8 个场景对象级地图已经具备可解释的语义标签。

### 3. 新增自动查询集生成器

新增代码：

```text
openfusion_agent/benchmark_generator.py
```

该模块从 `outputs/object_maps/replica_*_object_map.json` 自动生成查询样本，每条样本包含：

```text
query_id
scene
query
type
object_map
near_threshold
top_k
plan
expected_labels
expected_object_ids
expected_reference_ids
expected_reference_labels
annotation_source
review_status
```

当前生成的查询类型包括：

```text
category
functional
spatial_relation
```

其中空间关系查询会先执行一次结构化 QueryPlan，只保留当前对象地图中确实可以返回结果的样本。

### 4. 新增 160 条 Replica 自动候选查询集

新增数据文件：

```text
benchmarks/replica_auto_seed_queries.jsonl
```

查询集规模：

```text
总数: 160 queries
场景数: 8
每场景: 20 queries
category: 94
functional: 40
spatial_relation: 26
review_status: auto_unreviewed
```

该查询集目前定位为自动候选标注，用于开发验证和论文实验集初稿。正式实验前仍需人工复核。

### 5. 增强 benchmark runner

修改代码：

```text
openfusion_agent/benchmark.py
```

新增参数：

```text
--limit
```

用途：只运行前 N 条样本，方便进行 DeepSeek API 小样本冒烟测试，避免每次都调用完整 160 条查询。

## 三、实验验证结果

### 1. 固定 QueryPlan benchmark

运行命令：

```powershell
docker --context desktop-linux run --rm -v E:/OpenFusion:/workspace -w /workspace openfusion:local python -m openfusion_agent.benchmark --benchmark benchmarks/replica_auto_seed_queries.jsonl --output-dir outputs/benchmarks/replica_auto_seed --pretty
```

结果：

```text
num_cases: 160
has_result: 1.0
top1_label_match: 1.0
topk_label_match: 1.0
top1_object_match: 1.0
topk_object_match: 1.0
reference_id_match: 1.0
reference_label_match: 1.0
avg_latency_ms: 0.126
```

说明：结构化查询执行链路已可以稳定处理 160 条跨场景查询。

### 2. DeepSeek LLM 端到端冒烟测试

运行命令：

```powershell
docker --context desktop-linux run --rm -e DEEPSEEK_API_KEY=$env:DEEPSEEK_API_KEY -v E:/OpenFusion:/workspace -w /workspace openfusion:local python -m openfusion_agent.benchmark --benchmark benchmarks/replica_auto_seed_queries.jsonl --output-dir outputs/benchmarks/replica_auto_seed_llm20 --use-llm --labels "vase,table,tv shelf,curtain,wall,floor,ceiling,door,tv,room plant,light,sofa,cushion,wall paint,chair" --limit 20 --pretty
```

结果：

```text
num_cases: 20
has_result: 1.0
top1_label_match: 1.0
topk_label_match: 1.0
top1_object_match: 1.0
topk_object_match: 1.0
reference_id_match: 1.0
reference_label_match: 1.0
avg_latency_ms: 820.809
```

说明：DeepSeek 解析、QueryPlan schema、查询执行器和空间关系约束在小样本端到端链路中均可正常工作。

## 四、当前问题

1. 当前 160 条查询是自动生成的候选标注，不等同于人工确认的正式 benchmark。
2. 空间关系查询数量为 26 条，后续需要继续增加并人工筛选。
3. 目前只对前 20 条进行了 DeepSeek 端到端测试，完整 160 条 LLM 测试仍需在人工复核后运行。
4. 功能性查询仍存在概念边界问题，例如“装饰物”可能被 LLM 扩展到绿植、窗帘、靠垫等类别，因此查询文本需要进一步规范化。

## 五、下周计划

1. 增加人工审核脚本，对 `auto_unreviewed` 查询逐条确认。
2. 将 160 条自动候选查询整理为人工确认版 benchmark。
3. 扩展空间关系查询数量，重点覆盖 `near`、`closest_to`、`left_of`、`right_of`、`in_front_of`、`behind`。
4. 跑完整 160 条 DeepSeek 端到端 benchmark，并统计失败案例。
5. 将实验表格整理为论文可用格式。
