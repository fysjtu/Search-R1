# Search-R1 数据集分析

## 概述

Search-R1 主要使用 **RUC-NLPIR/FlashRAG_datasets** 作为数据源，这是一个在 HuggingFace 上公开的数据集集合。框架支持多种问答任务数据集，用于训练能够自主调用搜索引擎的 LLM。

---

## 支持的数据集列表

根据 `verl/trainer/main_ppo.py` 中的配置，框架支持以下数据集：

### 1. 开放域问答数据集

| 数据集名称 | 描述 | 用途 | 评分方式 |
|------------|------|------|----------|
| **NQ** | Natural Questions | 开放域问答 | Exact Match (EM) |
| **TriviaQA** | TriviaQA | 开放域问答 | Exact Match (EM) |
| **PopQA** | PopQA | 流行问答 | Exact Match (EM) |
| **HotpotQA** | 多跳推理问答 | 多跳问答 | Exact Match (EM) |
| **2WikiMultihopQA** | 2WikiMultihopQA | 多跳问答 | Exact Match (EM) |
| **MuSiQue** | Multi-Step Questions | 多跳问答 | Exact Match (EM) |
| **Bamboogle** | Bamboogle | 搜索问答 | Exact Match (EM) |

### 2. 数学推理数据集

| 数据集名称 | 描述 | 用途 | 评分方式 |
|------------|------|------|----------|
| **GSM8K** | Grade School Math | 数学推理 | 数值精确匹配 |

### 3. 其他数据集

| 数据集名称 | 描述 | 来源 |
|------------|------|------|
| **StrategyQA** | 策略推理问答 | 本地 JSON 文件 |

---

## 数据集详细说明

### 1. Natural Questions (NQ)

**文件**: `scripts/data_process/nq.py`, `scripts/data_process/nq_search.py`

**用途**: 
- 开放域问答任务
- 训练模型理解问题并生成答案

**数据来源**: `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')`

**Prompt 模板** (无搜索版本):
```
Answer the given question. 
You should first have a reasoning process in mind and then provides the answer. 
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. 
Question: {question}
```

**Prompt 模板** (搜索版本):
```
Answer the given question. 
You must conduct reasoning inside <think> and </think> first every time you get new information. 
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. 
You can search as many times as your want. 
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. 
Question: {question}
```

**数据结构**:
```python
{
    "question": "Who is the president?",
    "golden_answers": ["Joe Biden"],  # 可能有多个正确答案
    "context": "..."  # 可选上下文
}
```

---

### 2. RAG 版本数据集 (NQ + 检索上下文)

**文件**: `scripts/data_process/nq_rag.py`

**用途**: 
- 预检索版本：预先为每个问题检索相关文档
- 不调用搜索引擎，直接提供上下文进行问答

**特点**:
- 使用 E5 模型进行预检索
- 检索结果作为 prompt 的一部分
- 适合研究检索-生成范式

**Prompt 模板**:
```
Answer the given question with some potentially useful context. 
You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. 
You should first have a reasoning process in mind and then provides the answer. 
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. 
Question: {question} Context: {context} 
```

**检索缓存**:
```python
# 从预先计算的检索结果加载
retrieval_cache = {
    "question": [
        {"id": "doc_id", "score": 0.95},
        ...
    ]
}

# 然后从 corpus 中加载实际内容
corpus = {"doc_id": {"id": "...", "contents": "..."}}
```

---

### 3. 多数据集合并训练

**文件**: `scripts/data_process/qa_search_train_merge.py`, `scripts/data_process/qa_search_test_merge.py`

**用途**: 
- 合并多个数据集进行训练
- 提供更丰富的训练样本

**支持的合并方式**:
```bash
--data_sources nq,triviaqa,hotpotqa  # 用逗号分隔多个数据集
```

**特点**:
- 自动拼接多个数据集
- 保持数据源标签（`data_source`）
- 适用于混合训练场景

---

## 数据格式规范

### 标准数据格式

每个训练/测试样本应包含以下字段：

```python
{
    "data_source": "nq",  # 数据集标识
    "prompt": [{
        "role": "user",
        "content": "完整的问题 prompt"
    }],
    "ability": "fact-reasoning",  # 任务类型
    "reward_model": {
        "style": "rule",  # 奖励计算方式
        "ground_truth": {
            "target": ["正确答案1", "正确答案2"]  # 可能有多个标准答案
        }
    },
    "extra_info": {
        'split': 'train',  # 数据集划分
        'index': 0,  # 样本索引
    }
}
```

### 额外字段说明

- **`data_source`**: 用于在 `main_ppo.py` 中选择对应的评分函数
- **`reward_model.style`**: 
  - `"rule"`: 基于规则的评分（Exact Match）
  - 未来可扩展其他风格
- **`golden_answers`**: 支持多个正确答案的列表

---

## 奖励评分机制

### 1. Exact Match (EM) 评分

**文件**: `verl/utils/reward_score/qa_em.py`

**支持的函数**:
- `compute_score_em()`: 标准精确匹配
- `compute_score_subem()`: 子串精确匹配

**评分逻辑**:

```python
def compute_score_em(solution_str, ground_truth, format_score=0., score=1.):
    """
    从生成的文本中提取答案
    与标准答案进行标准化比较
    返回奖励分数
    """
    answer = extract_solution(solution_str)  # 提取 <answer>...</answer> 中的内容
    
    if answer is None:
        return 0  # 无有效答案
    elif em_check(answer, ground_truth['target']):
        return score  # 答案正确 (1.0)
    else:
        return format_score  # 格式正确但答案错误 (0.0)
```

**答案提取**:
```python
# 从生成文本中提取最后一个 <answer> 标签的内容
answer_pattern = r'<answer>(.*?)</answer>'
matches = re.finditer(answer_pattern, solution_str, re.DOTALL)
return matches[-1].group(1).strip()  # 返回最后一个匹配
```

**标准化**:
```python
def normalize_answer(s):
    # 1. 去除冠词 (a, an, the)
    # 2. 统一空格
    # 3. 去除标点
    # 4. 转小写
    return " ".join(text.split()).lower()
```

### 2. 格式化奖励评分

**文件**: `verl/utils/reward_score/qa_em_format.py`

**多层次奖励**:

| 情况 | 分数 | 说明 |
|------|------|------|
| 答案正确 + 格式正确 | 1.0 | 完美 |
| 答案正确 + 格式错误 | 0.8 | 轻微奖励 |
| 无答案但检索到相关信息 + 格式正确 | 0.3 | 部分奖励 |
| 无答案但检索到相关信息 + 格式错误 | 0.2 | 结构分 |
| 格式正确但检索错误 | 0.2 | 结构分 |
| 仅格式正确 | 0.1 | 格式分 |
| 其他 | 0 | 无奖励 |

**格式验证**:
```python
def is_valid_sequence(text):
    # 检查以下标签是否平衡：
    # - <think> ... </think>
    # - <search> ... </search>
    # - <information> ... </information>
    # - <answer> ... </answer>
    
    # 验证标签序列的合理性
    # 例如：reasoning -> search -> information -> reasoning -> answer
```

### 3. GSM8K 数学题评分

**文件**: `verl/utils/reward_score/gsm8k.py`

**提取方式**:
```python
# 从文本中提取最后一个 #### 数字
solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
final_answer = solution.group(0).split('#### ')[1]
```

**评分**:
```python
if answer == ground_truth:
    return 1.0  # 数值完全匹配
else:
    return 0.0
```

---

## 数据加载流程

### 1. 数据处理

```bash
# 处理单个数据集
python scripts/data_process/nq_search.py --local_dir ./data/nq_search

# 合并多个数据集
python scripts/data_process/qa_search_train_merge.py \
    --data_sources nq,triviaqa,hotpotqa \
    --local_dir ./data/qa_mixed
```

### 2. 数据转换

数据从 HuggingFace 加载后，经过以下处理：

```python
# 1. 标准化问题格式（确保以 ? 结尾）
example['question'] = example['question'].strip()
if example['question'][-1] != '?':
    example['question'] += '?'

# 2. 构建 prompt
question = make_prefix(example, template_type='base')

# 3. 构建数据项
data = {
    "data_source": "nq",
    "prompt": [{"role": "user", "content": question}],
    "reward_model": {"style": "rule", "ground_truth": {...}},
    ...
}

# 4. 保存为 parquet 格式
dataset.to_parquet('train.parquet')
```

### 3. 训练时加载

```python
# main_ppo.py 中的数据加载
data.train_files = './data/nq_search/train.parquet'
data.val_files = './data/nq_search/test.parquet'

# 根据 data_source 选择对应的评分函数
data_source = data_item.non_tensor_batch['data_source']
compute_score_fn = _select_rm_score_fn(data_source)  # 选择评分函数

# 计算奖励
score = compute_score_fn(solution_str=sequences_str, 
                         ground_truth=ground_truth)
```

---

## 语料库 (Corpus)

### Wiki-18 语料库

**用途**: 作为搜索引擎的检索目标

**格式** (JSONL):
```json
{"id": "0", "contents": "\"Evan Morris\"\nEvan L. Morris (January 26, 1977 – July 9, 2015) was a lobbyist..."}
{"id": "1", "contents": "\"Title\"\nContent text..."}
...
```

**特点**:
- 每行一个文档
- `id`: 文档唯一标识
- `contents`: 第一行为标题（用引号包裹），后面是正文

**索引**:
```bash
bash search_r1/search/build_index.sh
# 使用 BM25 或 Dense Retriever (E5/BGE) 构建索引
```

---

## 数据集使用示例

### 1. 训练 NQ 数据集

```bash
# 1. 处理数据
python scripts/data_process/nq_search.py --local_dir ./data/nq_search

# 2. 启动检索服务器
bash retrieval_launch.sh

# 3. 训练
bash train_ppo.sh
```

### 2. 混合数据集训练

```bash
# 合并多个数据集
python scripts/data_process/qa_search_train_merge.py \
    --data_sources nq,triviaqa,hotpotqa \
    --local_dir ./data/mixed_qa

# 更新 train_ppo.sh 中的数据路径
export DATA_DIR='data/mixed_qa'

# 训练
bash train_ppo.sh
```

### 3. 自定义数据集

要添加新的数据集，需要：

1. **添加数据源标识**（在 `main_ppo.py` 中）:
```python
def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'your_new_dataset', ...]:
        return qa_em.compute_score_em
```

2. **创建数据处理脚本**:
```python
# scripts/data_process/your_dataset.py
dataset = datasets.load_dataset('your_dataset_path')
# 转换为标准格式
```

3. **运行数据处理**:
```bash
python scripts/data_process/your_dataset.py
```

---

## 数据统计

### 数据量（估算）

| 数据集 | 训练集 | 验证集 | 来源 |
|--------|--------|--------|------|
| NQ | ~87K | 3K | Google |
| TriviaQA | ~95K | 11K | Trivia |
| HotpotQA | ~90K | 7K | Wikipedia |

### 数据特点

- **问题类型**: 事实性问答、多跳推理、数学计算
- **答案类型**: 短答案（短语）、数值、列表
- **难度**: 从简单到复杂的多层难度

---

## 总结

Search-R1 使用 **8+ 个问答数据集**，涵盖：
- ✅ 开放域问答 (NQ, TriviaQA, PopQA)
- ✅ 多跳推理 (HotpotQA, 2WikiMultihopQA, MuSiQue)
- ✅ 数学推理 (GSM8K)
- ✅ 策略推理 (StrategyQA)

所有数据集都通过 **RUC-NLPIR/FlashRAG_datasets** 在 HuggingFace 上统一加载和管理，并使用 **Exact Match** 作为主要的奖励评分方式。


