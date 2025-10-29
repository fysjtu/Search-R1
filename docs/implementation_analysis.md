# Search-R1 代码实现详细分析

## 项目概述

**Search-R1** 是一个基于强化学习（RL）的框架，用于训练能够自主调用搜索引擎进行多轮推理的大语言模型（LLM）。该框架基于 veRL（Volcano Engine Reinforcement Learning）构建，支持 PPO、GRPO 等强化学习算法。

---

## 整体架构

### 目录结构

```
Search-R1/
├── search_r1/              # 核心搜索与生成模块
│   ├── llm_agent/         # LLM 代理管理器
│   │   ├── generation.py  # 多轮生成核心逻辑
│   │   └── tensor_helper.py  # 张量处理辅助工具
│   └── search/            # 搜索引擎模块
│       ├── retrieval.py   # 检索器基类实现
│       └── retrieval_server.py  # FastAPI 检索服务
├── verl/                  # veRL 强化学习框架
│   ├── trainer/           # 训练器
│   │   ├── main_ppo.py    # PPO 训练主入口
│   │   └── ppo/
│   │       └── ray_trainer.py  # Ray 分布式训练器
│   ├── workers/           # 工作节点
│   └── utils/             # 工具函数
└── scripts/               # 数据处理脚本
```

---

## 核心模块详细分析

### 1. LLM 代理生成管理器 (`generation.py`)

#### 1.1 `LLMGenerationManager` 类

**功能**: 管理多轮对话生成，协调 LLM 与搜索引擎的交互

**关键配置** (`GenerationConfig`):

```python
@dataclass
class GenerationConfig:
    max_turns: int              # 最大对话轮数（默认2轮）
    max_start_length: int       # 初始 prompt 最大长度
    max_prompt_length: int      # prompt 最大总长度
    max_response_length: int    # 单轮响应最大长度
    max_obs_length: int         # 观察值（检索结果）最大长度
    num_gpus: int              # GPU 数量
    search_url: str             # 检索服务器 URL
    topk: int                   # 检索文档数量
```

#### 1.2 多轮生成流程 (`run_llm_loop`)

**核心循环逻辑**:

```python:220:319:search_r1/llm_agent/generation.py
def run_llm_loop(self, gen_batch, initial_input_ids):
    """
    主生成循环：
    1. 初始化状态
    2. 循环最多 max_turns 轮
       - 生成响应
       - 解析动作（search/answer）
       - 执行动作
       - 更新状态
    3. 最终生成答案
    """
```

**详细流程**:

1. **初始化状态**
   - `original_left_side`: 初始 prompt（用户问题）
   - `original_right_side`: 保存所有生成的 responses
   - `active_mask`: 标记哪些样本还在生成中

2. **循环生成** (最多 `max_turns` 轮):
   ```python
   for step in range(max_turns):
       # a. 生成响应
       responses = model.generate(prompt)
       
       # b. 解析动作
       action, content = parse_action(responses)  
       # action: 'search' 或 'answer'
       
       # c. 执行动作
       if action == 'search':
           search_results = call_search_engine(content)
           next_obs = format_search_results(search_results)
       elif action == 'answer':
           done = True
       
       # d. 更新状态
       prompt += response + next_obs  # 追加检索结果到 prompt
   ```

3. **状态管理**:
   - `_update_rolling_state()`: 合并 prompt + response + observation
   - `_info_masked_concatenate_with_padding()`: 创建 info_mask 屏蔽检索信息

#### 1.3 关键方法解析

##### `execute_predictions()` - 执行预测动作

```python:353:405:search_r1/llm_agent/generation.py
def execute_predictions(self, predictions, pad_token, active_mask, do_search=True):
    """
    执行预测动作：
    1. 解析 <search>query</search> 或 <answer>content</answer>
    2. 调用搜索引擎获取结果
    3. 返回观察值和终止标志
    """
    cur_actions, contents = self.postprocess_predictions(predictions)
    
    for action in cur_actions:
        if action == 'search':
            search_results = self.batch_search([query])
            next_obs = f'<information>{results}</information>'
        elif action == 'answer':
            done = True
```

**返回**:
- `next_obs`: 观察值（检索结果或空）
- `dones`: 是否终止
- `valid_action`: 动作是否有效
- `is_search`: 是否为搜索动作

##### `_info_masked_concatenate_with_padding()` - 信息屏蔽机制

```python:120:143:search_r1/llm_agent/generation.py
def _info_masked_concatenate_with_padding(self, prompt, prompt_with_mask, response, info=None):
    """
    创建两个版本：
    1. responses: 完整版本（包含检索信息）
    2. responses_with_info_mask: 屏蔽版本（info_mask = pad_id）
    
    目的：RL 训练时，奖励只针对 LLM 推理，不包括检索信息
    """
    if info is not None:
        # 创建检索信息的 mask（全为 pad_token_id）
        info_mask = torch.full(info.size(), pad_token_id, ...)
        
        # 将检索信息与响应拼接，但创建屏蔽版本
        tensors_with_mask.append(info_mask)
    
    # 返回两个版本
    return concatenated, concatenated_with_info
```

**目的**: 
- 奖励机制只关注 LLM 的推理过程
- 检索到的信息不被计入可学习部分
- 避免模型过度依赖检索质量

##### `_generate_with_gpu_padding()` - 多 GPU 批处理

```python:169:218:search_r1/llm_agent/generation.py
def _generate_with_gpu_padding(self, active_batch):
    """
    处理 batch_size 不是 num_gpus 倍数的情况：
    1. 计算 remainder = batch_size % num_gpus
    2. 用第一个序列填充 remainder 个样本
    3. 生成后移除填充
    """
```

**问题**: vLLM 要求 batch_size 必须是 num_gpus 的倍数

**解决方案**: 
- 检测 remainder
- 用第一个样本重复填充
- 生成后删除填充部分

---

### 2. 搜索引擎模块 (`search/`)

#### 2.1 `retrieval_server.py` - FastAPI 检索服务

**功能**: 提供 HTTP 检索接口，支持 BM25 和 Dense 检索

**API 端点**:

```python:326:358:search_r1/search/retrieval_server.py
@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    输入格式:
    {
      "queries": ["查询1", "查询2", ...],
      "topk": 3,
      "return_scores": true
    }
    """
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    return {"result": resp}
```

#### 2.2 检索器类型

##### BM25Retriever (稀疏检索)

```python:146:205:search_r1/search/retrieval_server.py
class BM25Retriever(BaseRetriever):
    """
    基于 Anserini 的 BM25 检索
    - 不需要 GPU
    - 速度快
    - 准确率相对较低
    """
    def _search(self, query, num, return_score):
        hits = self.searcher.search(query, num)
        # 返回文档列表
        return results
```

##### DenseRetriever (稠密检索)

```python:207:271:search_r1/search/retrieval_server.py
class DenseRetriever(BaseRetriever):
    """
    基于 FAISS 的 Dense 检索
    - 需要 GPU（可选）
    - 准确率高
    - 支持批处理
    """
    def _batch_search(self, query_list, num):
        # 编码查询
        query_emb = self.encoder.encode(query_list)
        
        # FAISS 搜索
        scores, idxs = self.index.search(query_emb, k=num)
        
        # 加载文档
        results = load_docs(self.corpus, idxs)
        return results
```

**支持的模型**:
- E5 (`intfloat/e5-base-v2`)
- BGE (`bge-large-en-v1.5`)
- DPR
- T5

**编码器处理**:

```python:90:105:search_r1/search/retrieval_server.py
if "e5" in self.model_name.lower():
    if is_query:
        query_list = [f"query: {query}" for query in query_list]
    else:
        query_list = [f"passage: {query}" for query in query_list]

if "bge" in self.model_name.lower():
    if is_query:
        query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]
```

---

### 3. veRL 强化学习框架

#### 3.1 PPO 训练主入口 (`main_ppo.py`)

```python:104:110:verl/trainer/main_ppo.py
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {...}})
    
    ray.get(main_task.remote(config))
```

**关键组件**:

1. **Role 定义**:
   ```python:48:59:verl/trainer/ppo/ray_trainer.py
   class Role(Enum):
       Actor = 0           # 策略网络
       Rollout = 1         # 推理生成
       ActorRollout = 2    # 统一 Actor+Rollout
       Critic = 3          # 价值网络
       RefPolicy = 4       # 参考策略
       RewardModel = 5     # 奖励模型
   ```

2. **资源池管理**:
   ```python:61:85:verl/trainer/ppo/ray_trainer.py
   class ResourcePoolManager:
       """
       管理 GPU 资源分配
       - 支持多节点训练
       - 动态负载均衡
       """
   ```

#### 3.2 RayPPOTrainer - PPO 训练器

**训练循环** (`fit` 方法):

```python:654:842:verl/trainer/ppo/ray_trainer.py
def fit(self):
    for epoch in range(total_epochs):
        for batch in train_dataloader:
            # 1. Rollout: 生成响应
            if do_search:
                final_gen_batch = generation_manager.run_llm_loop(...)
            else:
                gen_batch = actor_rollout_wg.generate_sequences(...)
            
            # 2. 计算参考策略的 log_prob
            ref_log_prob = ref_policy_wg.compute_ref_log_prob(batch)
            
            # 3. 计算价值估计
            values = critic_wg.compute_values(batch)
            
            # 4. 计算奖励
            reward_tensor = reward_fn(batch)  # Exact Match
            
            # 5. 应用 KL 惩罚
            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl)
            
            # 6. 计算优势函数 (GAE/GRPO)
            advantages = compute_advantage(batch, adv_estimator='gae')
            
            # 7. 更新 Critic
            critic_wg.update_critic(batch)
            
            # 8. 更新 Actor
            actor_rollout_wg.update_actor(batch)
```

#### 3.3 奖励计算 (`RewardManager`)

```python:32:97:verl/trainer/main_ppo.py
class RewardManager:
    def __call__(self, data: DataProto):
        """
        计算 Exact Match 奖励：
        1. 解码完整的序列 (prompt + response)
        2. 提取答案
        3. 与 ground truth 对比
        4. 返回 1.0 或 0.0
        """
        for i in range(len(data)):
            # 提取答案
            answer = extract_solution(sequences_str)
            
            # Exact Match 判断
            if em_check(answer, ground_truth):
                reward_tensor[i, -1] = 1.0
            else:
                reward_tensor[i, -1] = 0.0
        
        return reward_tensor
```

**奖励机制**:
- **Exact Match (EM)**: 标准化答案后进行字符串匹配
- **格式奖励**: 可选的对格式合规性的奖励
- **信息屏蔽**: 检索信息不参与奖励计算

---

### 4. 数据处理

#### 4.1 NQ 数据处理 (`nq_search.py`)

```python:60:84:scripts/data_process/nq_search.py
def process_fn(example, idx):
    question = example['question'].strip()
    if question[-1] != '?':
        question += '?'
    
    # 创建 prompt
    prompt = f"""Answer the given question. 
You must conduct reasoning inside <think>...</think> first.
After reasoning, if you find you lack some knowledge, you can call a search engine 
by <search> query </search>.
If you find no further external knowledge needed, you can directly provide the answer 
inside <answer> and </answer>. 
Question: {question}\n"""
    
    # 构建数据格式
    data = {
        "data_source": "nq",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": example['golden_answers']}
        }
    }
    return data
```

**数据格式**:
- `prompt`: 用户提示（包含问题）
- `reward_model`: 奖励配置（ground truth）
- `extra_info`: 元信息（split, index）

---

## 关键技术细节

### 1. 多轮对话状态管理

**挑战**: 需要在多轮对话中维护状态，同时支持批处理

**解决方案**:

```python:93:118:search_r1/llm_agent/generation.py
def _update_rolling_state(self, rollings, cur_responses, next_obs_ids):
    """
    更新滚动状态：
    1. 拼接 input_ids + responses + next_obs
    2. 创建 attention_mask 和 position_ids
    3. 截断到有效长度
    """
    new_input_ids = concatenate([rollings.input_ids, 
                                  cur_responses, 
                                  next_obs_ids])
    
    # 创建 attention mask
    new_attention_mask = create_attention_mask(new_input_ids)
    
    # 截断到有效长度
    effective_len = new_attention_mask.sum(dim=1).max()
    max_len = min(config.max_prompt_length, effective_len)
    
    new_rollings = DataProto.from_dict({
        'input_ids': new_input_ids[:, -max_len:],
        'attention_mask': new_attention_mask[:, -max_len:],
        'position_ids': position_ids[:, -max_len:]
    })
    return new_rollings
```

### 2. 信息屏蔽机制

**目的**: RL 训练时，奖励只针对 LLM 的推理，不包括检索信息

**实现**:

```python:120:143:search_r1/llm_agent/generation.py
def _info_masked_concatenate_with_padding(self, prompt, prompt_with_mask, response, info=None):
    """
    创建两个版本：
    1. responses: 完整版本（prompt + response + info）
    2. responses_with_info_mask: 屏蔽版本（info_mask = pad_token_id）
    """
    if info is not None:
        # 创建全 pad_token_id 的 info_mask
        info_mask = torch.full(info.size(), pad_token_id, ...)
        tensors_with_mask.append(info_mask)
    
    # 返回两个版本
    return concatenated, concatenated_with_info
```

**应用场景**:
- `responses`: 用于生成时提供上下文
- `responses_with_info_mask`: 用于 RL 训练时的奖励计算

### 3. Batch 搜索优化

**问题**: 多个样本同时执行搜索，需要批量处理

**解决方案**:

```python:438:458:search_r1/llm_agent/generation.py
def batch_search(self, queries: List[str]):
    """
    批量搜索优化：
    1. 收集所有 search action 的查询
    2. 一次性调用检索 API
    3. 按顺序分发结果
    """
    if do_search:
        search_results = self._batch_search(queries)
    else:
        search_results = [''] * len(queries)
    
    # 按顺序分发
    for action, content in zip(cur_actions, contents):
        if action == 'search':
            results = search_results.pop(0)
```

### 4. 多 GPU 训练优化

**FSDP 配置**:

```bash
# train_ppo.sh
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.grad_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
```

**优势**:
- 参数、梯度、优化器状态可卸载到 CPU
- 支持更大模型训练
- 减少显存占用

**序列长度平衡**:

```python:637:653:verl/trainer/ppo/ray_trainer.py
def _balance_batch(self, batch, metrics):
    """
    重新排序 batch，使每个 DP rank 的 token 数相似
    目的是提高训练效率
    """
    global_seqlen_lst = attention_mask.sum(-1).tolist()
    partitions = get_seqlen_balanced_partitions(
        global_seqlen_lst, 
        k_partitions=world_size
    )
    global_idx = torch.tensor([j for partition in partitions for j in partition])
    batch.reorder(global_idx)
```

---

## 训练配置参数

### 关键超参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_turns` | 最大对话轮数 | 2 |
| `max_prompt_length` | Prompt 最大长度 | 4096 |
| `max_response_length` | 单轮响应最大长度 | 500 |
| `max_start_length` | 初始 prompt 长度 | 2048 |
| `max_obs_length` | 观察值（检索结果）长度 | 500 |
| `kl_coef` | KL 散度惩罚系数 | 0.001 |
| `topk` | 检索文档数量 | 3 |
| `train_batch_size` | 训练批次大小 | 512 |
| `val_batch_size` | 验证批次大小 | 256 |

### PPO 配置

```bash
# train_ppo.sh
algorithm.adv_estimator=gae              # 优势估计方法：GAE 或 GRPO
algorithm.kl_ctrl.kl_coef=0.001          # KL 惩罚系数
algorithm.no_think_rl=false              # 是否屏蔽推理部分
actor_rollout_ref.actor.optim.lr=1e-6   # Actor 学习率
critic.optim.lr=1e-5                     # Critic 学习率
```

### 数据配置

```bash
data.train_files=$DATA_DIR/train.parquet
data.val_files=$DATA_DIR/test.parquet
data.train_batch_size=512
data.max_prompt_length=4096
```

---

## 完整训练流程

### 1. 数据准备

```bash
# 下载语料库
python scripts/download.py --save_path /path/to/save

# 处理 NQ 数据集
python scripts/data_process/nq_search.py
```

### 2. 启动检索服务

```bash
bash retrieval_launch.sh
# 启动 FastAPI 服务在 http://127.0.0.1:8000
```

### 3. 开始训练

```bash
bash train_ppo.sh
```

**训练过程**:

```
Epoch 1, Step 1:
  ├─ Rollout: 生成 512 个样本
  │  ├─ 第 1 轮: <search>query</search>
  │  ├─ 检索: 获取 top-3 文档
  │  └─ 第 2 轮: <answer>result</answer>
  ├─ 计算奖励: Exact Match
  ├─ 计算优势: GAE
  ├─ 更新 Critic
  └─ 更新 Actor

Epoch 1, Step 2:
  ...
```

### 4. 验证

```python:436:547:verl/trainer/ppo/ray_trainer.py
def _validate(self):
    """
    验证过程：
    1. 加载验证集
    2. 生成响应（使用 do_search=True）
    3. 计算 Exact Match 得分
    4. 记录指标
    """
    for batch in val_dataloader:
        # 多轮生成
        final_gen_batch = generation_manager.run_llm_loop(...)
        
        # 计算奖励
        reward_tensor = self.val_reward_fn(test_batch)
        
        # 记录指标
        metrics['val/test_score/nq'] = np.mean(rewards)
```

---

## 关键创新点

### 1. 多轮协作推理
- LLM 可以在需要时主动调用搜索引擎
- 动态的推理-搜索-推理循环
- 支持自主决定何时停止搜索

### 2. 信息屏蔽机制
- 检索信息不参与奖励计算
- 只奖励 LLM 的推理能力
- 避免过度依赖检索质量

### 3. 灵活的搜索引擎接入
- 支持本地检索器（BM25/Dense）
- 支持在线搜索 API（Google/Bing）
- 统一的 HTTP 接口

### 4. 高效的批处理
- 批量检索优化
- 多 GPU 自动填充
- 序列长度平衡

---

## 使用示例

### 推理

```python:94:129:infer.py
while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 生成（带停止条件）
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria
    )
    
    # 检查是否为 search 动作
    tmp_query = get_query(tokenizer.decode(outputs[0]))
    if tmp_query:
        search_results = search(tmp_query)
        
        # 拼接检索结果到 prompt
        search_text = f'<information>{search_results}</information>'
        prompt += search_text
    
    # 检查是否结束
    if outputs[0][-1].item() in [EOS_TOKEN]:
        break
```

### 自定义检索器

```python
# 1. 创建自定义检索器类
class CustomRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        # 初始化你的检索器
    
    def _search(self, query, num):
        # 实现单个查询搜索
        return results
    
    def _batch_search(self, query_list, num):
        # 实现批量搜索
        return results

# 2. 更新 get_retriever()
def get_retriever(config):
    if config.retrieval_method == "custom":
        return CustomRetriever(config)
```

---

## 总结

Search-R1 提供了一个完整的框架，用于训练能够自主调用搜索引擎的 LLM。

**核心优势**:
- ✅ 多轮协作推理
- ✅ 灵活的搜索引擎接入
- ✅ 高效的分布式训练
- ✅ 精心设计的奖励机制
- ✅ 信息屏蔽机制

**技术亮点**:
1. **多轮状态管理**: 复杂的状态维护和批处理优化
2. **信息屏蔽**: 区分推理和检索信息
3. **批处理优化**: 批量搜索和 GPU 填充
4. **序列平衡**: 动态负载均衡

**应用场景**:
- 开放域问答（Open-Domain QA）
- 研究助手
- 信息检索增强的对话系统
- 工具增强型 LLM

---

## 参考资源

- 论文: [Search-R1](https://arxiv.org/abs/2503.09516), [Empirical Study](https://arxiv.org/abs/2505.15117)
- 模型和数据: [HuggingFace Collections](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5)
- 代码仓库: [GitHub](https://github.com/PeterGriffinJin/Search-R1)
