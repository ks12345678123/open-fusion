# 长期任务指引：基于 Open-Fusion 的三维语义地图 LLM 查询 Agent

创建日期：2026-06-23

## 1. 任务定位

本任务面向毕业论文方向，目标是在已经跑通 Open-Fusion 的基础上，构建一个面向室内场景的开放词汇三维语义地图查询 Agent。

当前 Open-Fusion 主要完成：

```text
RGB-D 序列 -> 三维语义地图 -> 开放词汇点云查询
```

本长期任务希望进一步完成：

```text
Open-Fusion 输出结果
-> 对象级三维语义地图
-> DeepSeek LLM 查询 Agent
-> 自然语言到结构化 3D 查询
-> 空间关系推理
-> 查询结果高亮和实验评估
```

建议论文题目方向：

```text
基于开放词汇三维语义地图的室内场景自然语言查询 Agent 研究
```

或：

```text
面向机器人任务理解的对象级开放词汇三维语义地图与查询 Agent 方法研究
```

## 2. 当前基础进度

| 模块 | 状态 | 说明 |
|---|---:|---|
| Docker + GPU 环境 | 已完成 | Windows 11 + Docker Desktop + RTX 4080 SUPER 可用 |
| Open-Fusion 镜像 | 已完成 | `openfusion:local` 已构建 |
| SEEM 权重 | 已完成 | 模型可加载，tokenizer fallback 已修复 |
| ICL 复现 | 已完成 | `kt0` 完整 `vlfusion` 已跑通 |
| Replica 复现 | 已完成 | 8 个官方场景完整 2000 帧均已跑通 |
| Replica 可视化检查 | 已完成 | 已生成离线总览图，结果无明显异常 |
| DeepSeek API 测试 | 已完成 | `deepseek-v4-flash` 可正常返回 `OK` |
| 对象级地图 | 原型完成 | 已实现语义颜色分组 + DBSCAN 聚类，Replica 8 个场景均可生成对象候选 |
| 查询 Agent | 原型完成 | 已实现 DeepSeek API 调用、固定 prompt、JSON schema 校验和失败重试 |
| 查询执行器 | 原型完成 | 已实现 QueryPlan 到 object_map 的 top-k 查询、类别匹配和基础空间关系评分 |
| 实验基准 | 未开始 | 需要构造查询集和标注结果 |

## 3. 总体系统方案

系统分为五层：

```text
1. 数据与建图层
   Replica / ICL / 后续 ScanNet 或真实 RGB-D
   -> Open-Fusion
   -> color_pc.ply / semantic_pc.ply / color_mesh.glb

2. 对象地图层
   点云读取
   语义区域提取
   对象聚类
   bbox / centroid / size / label candidates

3. LLM Agent 层
   DeepSeek API
   自然语言解析
   类别扩展
   输出结构化 JSON 查询计划

4. 查询执行层
   类别匹配
   属性匹配
   空间关系计算
   候选排序

5. 可视化与评估层
   高亮点云
   查询结果截图
   query benchmark
   accuracy / success rate / latency
```

## 4. 预期创新点

### 4.1 对象级开放词汇三维语义地图

Open-Fusion 原始输出偏点云级语义查询。本任务进一步构建对象级地图，保存：

```text
object_id
scene_id
label_candidates
semantic_score
centroid_3d
bbox_min
bbox_max
bbox_size
point_count
source_ply
```

预期贡献：

```text
点云级语义结果 -> 对象级语义实体
```

这会让自然语言查询、空间关系判断和结果解释更稳定。

### 4.2 LLM 驱动的结构化 3D 查询解析

用 DeepSeek API 将自然语言问题解析成固定 JSON，而不是让 LLM 直接回答。

示例输入：

```text
找桌子旁边的椅子
```

示例输出：

```json
{
  "intent": "find_object",
  "target": ["chair"],
  "attributes": [],
  "relations": [
    {
      "type": "near",
      "object": "table"
    }
  ],
  "top_k": 3
}
```

预期贡献：

```text
自然语言 -> 可执行 3D 查询计划
```

### 4.3 功能语义扩展

LLM 用于将功能描述转换成开放词汇类别集合。

示例：

```text
可以坐的地方 -> chair / sofa / bench / stool
可以放东西的地方 -> table / desk / shelf / cabinet
照明设备 -> lamp / light
出口 -> door
```

预期贡献：

```text
类别词查询 -> 功能语义查询
```

### 4.4 空间关系 grounding

查询执行器在本地三维空间中计算关系，而不是依赖 LLM 猜测。

需要支持的关系：

```text
near
on
under
inside
left_of
right_of
in_front_of
behind
closest_to
between
```

预期贡献：

```text
语义检索 -> 语义 + 几何约束检索
```

## 5. 功能模块与代码区规划

建议新增代码目录：

```text
openfusion_agent/
  __init__.py
  config.py
  llm_parser.py
  schemas.py
  object_map.py
  query_executor.py
  spatial_relations.py
  visualizer.py
  benchmark.py
  demo_query.py
```

### 5.1 `llm_parser.py`

职责：

```text
调用 DeepSeek API
将自然语言问题解析成 JSON 查询计划
校验 JSON 字段
失败时重试或返回错误原因
```

输入：

```text
query: str
available_labels: list[str]
schema: dict
```

输出：

```json
{
  "intent": "find_object",
  "target": ["chair"],
  "attributes": [],
  "relations": [],
  "top_k": 3
}
```

实现要求：

```text
不要把 API key 写死在代码中
使用环境变量 DEEPSEEK_API_KEY
默认模型使用 deepseek-v4-flash
默认关闭 thinking
所有 prompt 固定并写入代码或配置文件
```

### 5.2 `schemas.py`

职责：

```text
定义 QueryPlan、Relation、ObjectCandidate 等数据结构
负责 JSON schema 和字段校验
```

建议字段：

```text
intent
target
attributes
relations
top_k
constraints
```

### 5.3 `object_map.py`

职责：

```text
读取 color_pc.ply / semantic_pc.ply
提取候选语义区域
执行点云聚类
生成对象级地图
保存为 JSON / NPZ / PLY
```

建议输出：

```text
replica_office0_object_map.json
replica_office0_objects.ply
```

### 5.4 `spatial_relations.py`

职责：

```text
实现三维空间关系判断
```

基础函数：

```text
distance(a, b)
is_near(a, b, threshold)
is_on(a, b)
is_under(a, b)
is_inside(a, b)
is_left_of(a, b, frame)
is_in_front_of(a, b, frame)
```

### 5.5 `query_executor.py`

职责：

```text
接收 QueryPlan
在 ObjectMap 中查找候选对象
执行空间关系过滤
计算排序分数
返回 top-k 结果
```

排序因素：

```text
语义相似度
空间关系满足程度
对象点数/体积置信度
距离约束
```

### 5.6 `visualizer.py`

职责：

```text
将查询结果高亮输出
生成 result.ply
生成 result.png
生成查询报告 JSON
```

输出示例：

```text
outputs/query_results/office0/query_001_result.ply
outputs/query_results/office0/query_001_result.png
outputs/query_results/office0/query_001_result.json
```

### 5.7 `benchmark.py`

职责：

```text
批量运行查询集
统计成功率、Top-k accuracy、空间约束满足率、响应时间
生成表格
```

## 6. 数据与实验设计

### 6.1 数据集

第一阶段使用已完成复现的 Replica 8 个场景：

```text
replica_office0
replica_office1
replica_office2
replica_office3
replica_office4
replica_room0
replica_room1
replica_room2
```

后续扩展：

```text
ScanNet：用于论文级定量评估，受数据权限影响
TUM RGB-D / 7-Scenes：用于真实 RGB-D 定性测试
```

### 6.2 查询集设计

建议构造 160 到 240 条查询：

```text
每个场景 20 到 30 条
8 个场景共 160 到 240 条
```

查询类型：

| 类型 | 示例 | 目标 |
|---|---|---|
| 类别查询 | 找椅子 | 测试基础开放词汇检索 |
| 功能查询 | 找可以坐的地方 | 测试 LLM 类别扩展 |
| 空间关系查询 | 找桌子旁边的椅子 | 测试几何关系 grounding |
| 组合查询 | 找离门最近的可以坐的地方 | 测试复杂查询 |
| 否定/失败查询 | 找不存在的冰箱 | 测试鲁棒性 |

### 6.3 标注格式

建议建立：

```text
benchmarks/replica_queries.jsonl
```

每行一个 query：

```json
{
  "query_id": "office0_001",
  "scene": "replica_office0",
  "query": "找桌子旁边的椅子",
  "type": "spatial_relation",
  "expected_labels": ["chair"],
  "reference_objects": ["table"],
  "success_criteria": "top3_contains_correct_object",
  "notes": "人工检查可接受"
}
```

## 7. 评价指标

### 7.1 查询解析指标

```text
JSON valid rate
字段完整率
类别扩展准确率
平均 API 响应时间
```

### 7.2 查询结果指标

```text
Top-1 accuracy
Top-3 accuracy
Query success rate
Spatial constraint satisfaction rate
Failure rate
Average query latency
```

### 7.3 消融实验

至少比较以下版本：

```text
A. Open-Fusion 原始关键词查询
B. LLM 解析 + 点云级查询
C. LLM 解析 + 对象级地图
D. LLM 解析 + 对象级地图 + 空间关系约束
```

预期分析：

```text
B 相比 A：自然语言泛化能力提升
C 相比 B：定位和解释稳定性提升
D 相比 C：复杂空间查询准确率提升
```

## 8. 阶段计划

### 第 1 阶段：对象级地图原型

目标：

```text
从 Replica 输出点云生成 object map
```

任务：

- [x] 读取 `semantic_pc.ply`
- [x] 设计对象数据结构
- [x] 实现基础点云聚类
- [x] 计算 centroid / bbox / point_count
- [x] 保存 object map JSON
- [x] 输出对象可视化 PLY

验收标准：

```text
至少在 replica_office0 上生成 10 个以上可解释对象候选
```

### 第 2 阶段：DeepSeek 查询 Agent

目标：

```text
自然语言问题 -> JSON 查询计划
```

任务：

- [x] 封装 DeepSeek API 调用
- [x] 设计固定 prompt
- [x] 设计 JSON schema
- [x] 实现 JSON 校验
- [x] 测试类别查询、功能查询、空间关系查询

验收标准：

```text
20 条测试 query 中至少 18 条能输出合法 JSON
```

### 第 3 阶段：查询执行器

目标：

```text
QueryPlan -> 3D object candidates
```

任务：

- [x] 实现类别匹配
- [x] 实现功能类别扩展
- [x] 实现 near / closest_to
- [x] 实现 on / under 初版
- [x] 实现 top-k 排序
- [x] 输出查询结果 JSON

验收标准：

```text
能完成“找桌子旁边的椅子”等组合查询
```

### 第 4 阶段：可视化系统

目标：

```text
查询结果可展示
```

任务：

- [ ] 高亮目标对象点云
- [ ] 输出 result.ply
- [ ] 输出 result.png
- [ ] 生成查询结果报告

验收标准：

```text
每条 query 都能生成可视化结果和结构化日志
```

### 第 5 阶段：实验集与消融实验

目标：

```text
完成论文实验数据
```

任务：

- [ ] 构建 160 条以上 Replica 查询集
- [ ] 人工标注期望结果
- [ ] 跑 baseline
- [ ] 跑消融实验
- [ ] 统计指标
- [ ] 分析失败案例

验收标准：

```text
形成至少 3 张实验表格和 1 组可视化对比图
```

### 第 6 阶段：论文材料整理

目标：

```text
支撑毕业论文撰写
```

任务：

- [ ] 系统架构图
- [ ] 方法流程图
- [ ] 对象地图示意图
- [ ] 查询结果可视化图
- [ ] 实验表格
- [ ] 失败案例分析

## 9. 每周进度填写模板

每周更新时复制以下模板。

```markdown
## 第 X 周进度：YYYY-MM-DD 至 YYYY-MM-DD

### 本周目标

- [ ] 目标 1
- [ ] 目标 2
- [ ] 目标 3

### 已完成工作

1. 
2. 
3. 

### 代码变更

| 文件/目录 | 变更说明 | 状态 |
|---|---|---|
| `openfusion_agent/xxx.py` |  |  |

### 实验结果

| 实验 | 数据集/场景 | 指标 | 结果 |
|---|---|---:|---:|
|  |  |  |  |

### 当前问题

1. 
2. 
3. 

### 下周计划

- [ ] 
- [ ] 
- [ ] 

### 是否影响总体进度

```text
不影响 / 轻微影响 / 明显影响
```
```

## 10. 代码提交要求

建议每完成一个功能点提交一次。

提交信息建议：

```text
Add object map builder prototype
Add DeepSeek query parser
Add spatial relation executor
Add query visualization outputs
Add Replica query benchmark
```

不要提交：

```text
API key
.env 文件
大型点云结果
大型日志
Docker 缓存
数据集原始文件
```

需要加入 `.gitignore` 的内容：

```text
.env
outputs/query_results/
benchmarks/cache/
*_object_map.json
*_objects.ply
```

## 11. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 对象聚类不稳定 | 查询结果不稳定 | 先做粗粒度对象，后续再优化实例分割 |
| LLM 输出 JSON 不稳定 | 查询执行失败 | 固定 schema，增加校验和重试 |
| 空间关系定义模糊 | 评估困难 | 先只做 near / closest_to / on 三类关系 |
| Replica 缺少直接对象真值 | 定量评估难 | 先做人工 query benchmark，再研究 Replica semantic ground truth |
| ScanNet 权限问题 | 论文指标受限 | 把 ScanNet 作为扩展，不作为主线依赖 |

## 12. 当前下一步

建议立刻开始：

```text
第 4 阶段：可视化系统
```

最小可行目标：

```text
输入：QueryPlan 查询结果 + object_map.json + objects.ply
输出：高亮目标对象的 result.ply / result.png
记录：query_result.json
```

完成这个后，就可以形成“自然语言 -> LLM 解析 -> 3D 对象查询 -> 可视化结果”的最小闭环。

## 13. 开发记录

### 2026-06-23：对象级地图原型

已新增代码：

```text
openfusion_agent/__init__.py
openfusion_agent/config.py
openfusion_agent/object_map.py
```

已修改：

```text
main.py
.gitignore
```

主要功能：

```text
1. 从 Open-Fusion 的 semantic_pc.ply 读取语义点云。
2. 按语义颜色进行分组。
3. 在每个语义颜色分组内执行 Open3D DBSCAN 聚类。
4. 为每个对象候选计算 centroid、bbox、bbox_size、point_count。
5. 输出 object_map.json 和 objects.ply。
6. main.py 后续运行时会保存 semantic_label_map.json，避免语义颜色和类别名丢失。
```

验证命令：

```powershell
docker --context desktop-linux run --rm -v E:/OpenFusion:/workspace -w /workspace openfusion:local python -m openfusion_agent.object_map --scene-dir replica_office0 --output-dir outputs/object_maps --scene-name replica_office0
```

批量验证结果：

```text
replica_office0: 26 objects
replica_office1: 21 objects
replica_office2: 26 objects
replica_office3: 38 objects
replica_office4: 24 objects
replica_room0:   37 objects
replica_room1:   27 objects
replica_room2:   22 objects
```

当前限制：

```text
旧的 semantic_pc.ply 没有保存颜色到类别名的映射，因此当前对象标签暂时显示为 semantic_color_000 等占位名称。
后续重新运行 main.py 后，会生成 semantic_label_map.json，新对象地图即可恢复真实类别名。
```

下一步：

```text
开始第 4 阶段：可视化系统。
目标是把查询结果中的 object_id 高亮输出为 result.ply 和 result.png。
```

### 2026-06-23：查询执行器原型

已新增代码：

```text
openfusion_agent/spatial_relations.py
openfusion_agent/query_executor.py
```

主要功能：

```text
1. 读取 object_map.json。
2. 接收 QueryPlan。
3. 对 object semantic_label 做类别/同义词匹配。
4. 支持 near、closest_to、on、under、inside、left_of、right_of、in_front_of、behind 的基础评分。
5. 输出 top-k JSON 查询结果。
```

验证方式：

```text
当前旧 Replica object_map 尚无真实类别名，因此使用 semantic_color_000 等占位标签测试执行逻辑。
```

测试结果：

```text
类别查询 semantic_color_000：返回 4 个候选，top-3 正常排序。
closest_to semantic_color_001：返回 4 个候选，并包含 reference_object_id 和 distance。
near semantic_color_001：默认 1.5m 阈值下无满足候选，行为符合过滤逻辑。
```

当前限制：

```text
已有 Replica 输出是在保存 semantic_label_map.json 之前生成的，因此 object_map 中暂时没有 chair/table/sofa 等真实语义名。
后续若需要真实自然语言闭环，需要重新运行至少一个场景，生成 semantic_label_map.json 后再构建 object_map。
```

下一步：

```text
开始第 4 阶段：可视化系统。
目标是把查询结果中的 object_id 高亮输出为 result.ply 和 result.png。
```

### 2026-06-23：DeepSeek 查询 Agent 原型

已新增代码：

```text
openfusion_agent/schemas.py
openfusion_agent/llm_parser.py
```

主要功能：

```text
1. 使用 Python 标准库 urllib 调用 DeepSeek API，不依赖 OpenAI SDK。
2. 使用环境变量 DEEPSEEK_API_KEY，避免将 API key 写入代码。
3. 默认模型为 deepseek-v4-flash。
4. 默认关闭 thinking，降低输出为空或格式不稳定的概率。
5. 将自然语言查询解析为 QueryPlan JSON。
6. 支持 JSON markdown 包裹清理、schema 校验、top_k 限制和一次自动修复重试。
```

QueryPlan 格式：

```json
{
  "intent": "find_object",
  "target": ["chair"],
  "attributes": [],
  "relations": [
    {
      "type": "near",
      "object": "table"
    }
  ],
  "top_k": 3
}
```

验证命令：

```powershell
docker --context desktop-linux run --rm -e DEEPSEEK_API_KEY=$env:DEEPSEEK_API_KEY -v E:/OpenFusion:/workspace -w /workspace openfusion:local python -m openfusion_agent.llm_parser --query "找桌子旁边的椅子" --labels "chair,table,sofa,door,light,tv,cabinet,shelf" --top-k 3 --pretty
```

真实 API 测试结果：

```text
找桌子旁边的椅子
-> target: chair
-> relation: near table

找一个可以坐的地方
-> target: chair / sofa / bench / stool

找离门最近的灯
-> target: light / lamp
-> relation: closest_to door

找出口
-> target: door

找电视前面的沙发
-> target: sofa
-> relation: in_front_of tv
```

下一步：

```text
开始第 4 阶段：可视化系统。
目标是把查询结果中的 object_id 高亮输出为 result.ply 和 result.png。
```
