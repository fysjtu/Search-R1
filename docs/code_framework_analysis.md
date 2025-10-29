# Search-R1 代码框架分析

## 项目概述

**Search-R1** 是一个基于强化学习的框架，用于训练融合推理和搜索引擎调用的语言模型。项目基于 **veRL (Volcano Engine Reinforcement Learning)** 构建，目标是通过强化学习让大语言模型学习如何合理使用搜索引擎进行多轮推理。

---

## 整体架构

```
Search-R1
├── search_r1/              # 搜索相关核心模块
│   ├── llm_agent/          # LLM 代理生成管理器
│   └── search/              # 搜索引擎服务模块
├── verl/                    # veRL 强化学习训练框架
│   ├── trainer/             # 训练相关代码
│   ├── workers/             # 工作节点管理
│   └── utils/               # 工具函数
├── scripts/                 # 数据处理脚本
├── example/                 # 示例数据
└── docs/                    # 文档
```

---

## 核心模块详解

### 1. 搜索模块 (`search_r1/search/`)

#### 1.1 `retrieval_server.py`
- **功能**: 本地检索服务器（支持 BM25 和 Dense 检索）
- **主要类**:
  - `Encoder`: 编码器类，负责将查询和文档编码为向量
    - 支持多种模型：E5、BGE、T5、DPR 等
    - 提供 mean/CLS/pooler 多种池化方法
  - `BaseRetriever`: 抽象检索基类
  - `BM25Retriever`: BM25 稀疏检索器
  - `DenseRetriever`: 基于 FAISS 的稠密检索器
- **API**: 提供 `/retrieve` 端点，支持批量检索

#### 1.2 `retrieval_rerank_server.py`
- **功能**: 检索+重排序一体化服务
- **流程**: 检索 → 重排序 → 返回 top-k 结果
- **重排序模型**: 使用 Sentence-Transformers 的 CrossEncoder

---

### 2. LLM 代理模块 (`search_r1/llm_agent/`)

#### 2.1 `generation.py` - 核心生成管理器

**主要类**: `LLMGenerationManager`

**核心功能**:

```python
class LLMGenerationManager:
    def __init__(self, tokenizer, actor_rollout_wg, config, is_validation):
        # 初始化生成器配置
        # 设置多轮对话参数
```

**多轮对话流程** (`run_llm_loop`):

1. **初始化**: 设置初始输入和状态
2. **循环生成** (最多 `max_turns` 轮):
   - 调用 LLM 生成响应
   - 解析动作（search/answer）
   - 执行动作（调用搜索引擎或生成答案）
   - 更新状态
3. **最终生成**: 最后一轮生成最终答案

**关键方法**:

- `execute_predictions()`: 执行预测动作
  - 解析 `<search>query</search>` 和 `<answer>content</answer>` 标签
  - 调用搜索引擎获取结果
  - 返回观察值和终止标志

- `_update_rolling_state()`: 更新滚动状态
  - 合并 prompt + response + observation
  - 处理填充和截断

- `_info_masked_concatenate_with_padding()`: 
  - 特殊处理：创建 info_mask 用于屏蔽检索到的信息块
  - 这样 RL 训练时奖励只针对 LLM 的推理，不包括检索到的内容

**数据结构**:
- `left_side`: 初始 prompt
- `right_side`: 所有生成的 responses（包括推理和答案）
- `responses_with_info_mask`: 用于屏蔽检索信息的版本

---

### 3. veRL 训练框架 (`verl/`)

#### 3.1 训练器 (`verl/trainer/main_ppo.py`)

**PPO 训练主流程**:

```python
def train():
    # 1. 初始化角色和工作组
    # 2. 构建数据加载器
    # 3. 主训练循环
    for epoch in range(total_epochs):
        for step in range(training_steps):
            # a. Rollout 阶段：生成响应
            # b. 计算奖励
            # c. PPO 更新
```

**关键组件**:

- **Role**: 定义不同角色
  - `Actor`: 策略网络
  - `Rollout`: 推理生成
  - `Critic`: 价值网络
  - `RefPolicy`: 参考策略（用于 KL 散度计算）

- **RewardManager**: 奖励管理器
  - 计算答案准确率（Exact Match）
  - 支持不同的数据集（NQ, TriviaQA, HotpotQA 等）

- **核心算法**:
  - `apply_kl_penalty()`: 应用 KL 惩罚
  - `compute_advantage()`: 计算优势函数（GAE 或 GRPO）

#### 3.2 工作节点 (`verl/workers/`)

**Actor 工作节点** (`workers/actor/`):
- 策略网络训练
- 支持 FSDP 和 Megatron 两种并行策略

**Rollout 节点** (`workers/rollout/`):
- 使用 vLLM 进行高效推理生成
- 支持批量生成和流式输出

**Critic 节点** (`workers/critic/`):
- 价值网络估计
- 用于 GAE 优势计算

---

### 4. 数据处理 (`scripts/data_process/`)

#### `nq_search.py`
- **功能**: 处理 Natural Questions (NQ) 数据集
- **输出格式**:
```python
data = {
    "data_source": "nq",
    "prompt": [{"role": "user", "content": question}],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": answer
    },
    "extra_info": {...}
}
```

---

## 训练流程

### 1. 启动检索服务器

```bash
# 启动本地 BM25/E5 检索服务
bash retrieval_launch.sh
```

### 2. PPO 训练

```bash
bash train_ppo.sh
```

**训练参数** (关键配置):

```bash
# 数据配置
data.train_batch_size=512
data.max_prompt_length=4096
data.max_response_length=500
data.max_turns=2  # 最多 2 轮搜索+推理

# 模型配置
actor_rollout_ref.model.path=Llama-3.2-3B
actor_rollout_ref.actor.optim.lr=1e-6

# PPO 配置
algorithm.kl_ctrl.kl_coef=0.001  # KL 惩罚系数
algorithm.no_think_rl=false  # 是否屏蔽推理部分

# 搜索引擎配置
retriever.url="http://127.0.0.1:8000/retrieve"
retriever.topk=3  # 检索 top-3 文档
```

### 3. 训练循环（多轮交互）

```
Epoch 1:
  ┌─────────────────────────────────────┐
  │  用户问题: Who is the president?    │
  ├─────────────────────────────────────┤
  │  LLM 推理: 需要查找现任总统信息      │
  │  生成动作: <search>2024 US president</search> │
  ├─────────────────────────────────────┤
  │  搜索引擎返回: {返回检索结果}         │
  ├─────────────────────────────────────┤
  │  LLM 推理: 基于检索信息分析          │
  │  生成动作: <answer>Joe Biden</answer> │
  ├─────────────────────────────────────┤
  │  奖励计算: Exact Match = 1.0        │
  └─────────────────────────────────────┘
```

---

## 关键技术细节

### 1. 多轮对话状态管理

LLM 在每一轮都会：
1. 基于当前 prompt + 历史对话生成响应
2. 解析响应中的 action（search/answer）
3. 如果是 search，调用搜索引擎获取新信息
4. 将检索结果作为 observation 追加到 prompt
5. 下一轮继续基于更新后的 prompt 生成

### 2. 奖励机制

```python
def compute_reward(response, ground_truth):
    # Exact Match 评分
    return 1.0 if EM_match(response, ground_truth) else 0.0
```

- **规则奖励**: 基于答案准确率的二元奖励
- **格式奖励**: 可选的对格式合规性的额外奖励
- **屏蔽机制**: 检索到的信息被 info_mask 屏蔽，奖励只针对 LLM 的推理和答案部分

### 3. 信息屏蔽机制 (Info Masking)

```python
# 创建两个版本的 responses
responses              # 完整版本（用于推理）
responses_with_info    # 包含检索信息（用于上下文）
responses_with_info_mask  # 屏蔽版本（info_mask = 0，用于奖励计算）
```

**目的**: 
- 奖励机制只关注 LLM 的推理过程
- 检索到的信息不作为可学习部分
- 避免过度依赖检索质量

### 4. 批处理优化

- **多 GPU 填充**: 自动处理 batch_size 不是 num_gpus 倍数的情况
- **序列长度平衡**: 自动切割过长序列
- **记忆优化**: 使用 FSDP 的 param/grad/optimizer offload

---

## 配置参数说明

### 关键超参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_turns` | 最大对话轮数 | 2 |
| `max_prompt_length` | Prompt 最大长度 | 4096 |
| `max_response_length` | 单轮响应最大长度 | 500 |
| `max_start_length` | 初始 prompt 最大长度 | 2048 |
| `max_obs_length` | 观察值（检索结果）最大长度 | 500 |
| `kl_coef` | KL 散度惩罚系数 | 0.001 |
| `topk` | 检索文档数量 | 3 |

### 训练优化策略

```bash
# FSDP 配置
fsdp_config.param_offload=true      # 参数卸载到 CPU
fsdp_config.grad_offload=true        # 梯度卸载到 CPU
fsdp_config.optimizer_offload=true   # 优化器状态卸载
```

**好处**: 
- 支持更大模型训练
- 减少显存占用
- 提高训练稳定性

---

## 数据流

```
原始问题 
    ↓
[LLM 推理] → <search>query</search>
    ↓
[搜索引擎] → 检索结果
    ↓
[LLM 推理] → <answer>答案</answer>
    ↓
[奖励计算] → 准确率评分
    ↓
[PPO 更新] → 策略优化
```

---

## 支持的 RL 算法

1. **PPO**: Proximal Policy Optimization
   - 使用 GAE (Generalized Advantage Estimation)
   - KL 散度约束

2. **GRPO**: Group Relative Policy Optimization
   - 基于组内相对性能的奖励

3. **Reinforce**: 基线强化学习

---

## 支持的检索器

| 类型 | 说明 | 模型示例 |
|------|------|---------|
| **BM25** | 稀疏检索 | Anserini BM25 |
| **Dense** | 稠密检索 | E5, BGE, DPR |
| **Online** | 在线搜索 | Google, Bing, Brave |

---

## 推理使用

```python
# infer.py
question = "Who is the president of the US?"

# 构建 prompt，指导模型使用 <search> 和 <answer> 标签
prompt = f"Answer the given question. You must reason... Question: {question}"

# 多轮生成
while True:
    outputs = model.generate(...)
    if <search> 标签:
        # 调用搜索引擎
        search_results = search(query)
        prompt += f"<information>{search_results}</information>"
    elif <answer> 标签:
        # 返回最终答案
        break
```

---

## 扩展性

### 添加新的搜索引擎

1. 在 `search_r1/search/` 下创建新的 retriever 类
2. 继承 `BaseRetriever`
3. 实现 `_search()` 和 `_batch_search()` 方法
4. 更新 `get_retriever()` 函数

### 添加新的 RL 算法

1. 在 `verl/trainer/ppo/` 下添加新算法
2. 实现相应的计算逻辑
3. 在主训练循环中调用

### 支持新的数据集

1. 在 `scripts/data_process/` 下添加数据处理脚本
2. 在 `verl/utils/reward_score/` 下添加对应的评分函数
3. 更新数据加载配置

---

## 总结

Search-R1 提供了一个完整的框架，用于训练能够自主调用搜索引擎的 LLM。

**核心优势**:
- ✅ 多轮交互推理
- 灵活的搜索引擎接入
- 高效的分布式训练
- 完整的奖励机制

**应用场景**:
- 知识问答（Open-Domain QA）
- 研究助手
- 信息检索增强的对话系统

**技术亮点**:
1. **多轮协作**: LLM + 搜索引擎无缝协作
2. **状态管理**: 复杂的多轮状态维护
3. **信息屏蔽**: 精心设计的奖励机制
4. **高效训练**: 基于 veRL 的分布式优化

这个框架为探索工具增强型 LLM 提供了一个强大的基础平台。


