# Search-R1 数据下载与准备指南

## 概述

Search-R1 需要下载两类数据：
1. **训练数据集**：从 HuggingFace 自动加载（无需手动下载）
2. **检索语料库和索引**：需要手动下载

---

## 一、训练数据集（自动加载）

### 1.1 数据集来源

所有训练数据集都来自 HuggingFace 的 **RUC-NLPIR/FlashRAG_datasets**：

| 数据集 | 自动加载方式 |
|--------|------------|
| NQ | `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')` |
| TriviaQA | `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'triviaqa')` |
| HotpotQA | `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'hotpotqa')` |
| PopQA | `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'popqa')` |
| 2WikiMultihopQA | `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', '2wikimultihopqa')` |
| MuSiQue | `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'musique')` |
| Bamboogle | `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'bamboogle')` |

### 1.2 数据处理

运行数据处理脚本时，数据会自动从 HuggingFace 下载并缓存在本地：

```bash
# 处理 NQ 数据集（搜索版本）
python scripts/data_process/nq_search.py --local_dir ./data/nq_search

# 处理 NQ 数据集（RAG 版本）
python scripts/data_process/nq_rag.py --local_dir ./data/nq_rag

# 合并多个数据集
python scripts/data_process/qa_search_train_merge.py \
    --local_dir ./data/mixed \
    --data_sources nq,triviaqa,hotpotqa
```

**输出格式**: Parquet 文件（`train.parquet`, `test.parquet`）

---

## 二、检索语料库和索引（手动下载）

### 2.1 需要下载的内容

为了运行检索服务器，需要下载：

1. **Wiki-18 语料库** (`wiki-18.jsonl.gz`)
   - 约 18M 文档的维基百科数据
   - 用作检索目标库

2. **E5 索引文件** (`part_aa`, `part_ab`)
   - 使用 E5 模型构建的向量索引
   - 用于稠密检索

3. **可选：BM25 索引**
   - 使用 Anserini 构建的 BM25 索引
   - 用于稀疏检索

### 2.2 下载方式

#### 方法一：使用下载脚本（推荐）

```bash
# 设置保存路径
save_path=/path/to/save

# 运行下载脚本
python scripts/download.py --save_path $save_path

# 合并索引文件
cat $save_path/part_* > $save_path/e5_Flat.index

# 解压语料库
gzip -d $save_path/wiki-18.jsonl.gz
```

**下载内容**：
- `part_aa`, `part_ab` → 合并为 `e5_Flat.index`
- `wiki-18.jsonl.gz` → 解压为 `wiki-18.jsonl`

#### 方法二：手动下载

从 HuggingFace 手动下载：

**1. 下载 E5 索引**：
```bash
# 使用 huggingface_hub
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='PeterJinGo/wiki-18-e5-index', \
filename='part_aa', repo_type='dataset', local_dir='/path/to/save')"

python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='PeterJinGo/wiki-18-e5-index', \
filename='part_ab', repo_type='dataset', local_dir='/path/to/save')"

# 合并索引
cat /path/to/save/part_* > /path/to/save/e5_Flat.index
```

**2. 下载语料库**：
```bash
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='PeterJinGo/wiki-18-corpus', \
filename='wiki-18.jsonl.gz', repo_type='dataset', local_dir='/path/to/save')"

# 解压
gzip -d /path/to/save/wiki-18.jsonl.gz
```

**3. 下载链接**：
- 索引仓库：https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index
- 语料库：https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus

---

## 三、完整数据准备流程

### 场景一：快速开始（使用 NQ 数据集 + E5 检索器）

```bash
# 1. 下载检索索引和语料库
save_path=/path/to/data
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz

# 2. 处理训练数据（自动从 HuggingFace 下载 NQ）
python scripts/data_process/nq_search.py --local_dir ./data/nq_search

# 3. 启动检索服务器
bash retrieval_launch.sh

# 4. 开始训练
bash train_ppo.sh
```

### 场景二：使用 BM25 检索器

```bash
# 1. 下载语料库
save_path=/path/to/data
python scripts/download.py --save_path $save_path
gzip -d $save_path/wiki-18.jsonl.gz

# 2. 构建 BM25 索引
bash search_r1/search/build_index.sh \
    --retriever_name bm25 \
    --corpus_path $save_path/wiki-18.jsonl \
    --index_path ./index/bm25

# 3. 更新检索服务器配置（retrieval_launch.sh）
# 使用 BM25 相关的配置

# 4. 处理训练数据
python scripts/data_process/nq_search.py --local_dir ./data/nq_search

# 5. 启动服务器并训练
bash retrieval_launch.sh
bash train_ppo.sh
```

### 场景三：多数据集训练

```bash
# 1. 下载检索数据（同场景一）
save_path=/path/to/data
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz

# 2. 处理多个数据集
python scripts/data_process/qa_search_train_merge.py \
    --local_dir ./data/mixed_qa \
    --data_sources nq,triviaqa,hotpotqa

# 3. 更新训练脚本中的数据路径
export DATA_DIR='./data/mixed_qa'

# 4. 训练
bash train_ppo.sh
```

---

## 四、数据文件说明

### 4.1 训练数据格式

处理后的训练数据以 **Parquet** 格式保存：

```bash
data/
  nq_search/
    ├── train.parquet      # 训练集
    └── test.parquet        # 验证集
```

**示例数据内容**：
```python
{
    "data_source": "nq",
    "prompt": [{"role": "user", "content": "Question: Who is the president?"}],
    "reward_model": {"style": "rule", "ground_truth": {"target": ["Joe Biden"]}},
    "extra_info": {"split": "train", "index": 0}
}
```

### 4.2 语料库格式

语料库以 **JSONL** 格式保存，每行一个文档：

```bash
data/
  wiki-18/
    └── wiki-18.jsonl      # 解压后的语料库
```

**示例内容**：
```json
{"id": "0", "contents": "\"Document Title\"\nDocument content text..."}
{"id": "1", "contents": "\"Another Title\"\nAnother content..."}
```

### 4.3 索引文件

索引文件用于快速检索：

```bash
index/
  ├── e5_Flat.index      # E5 稠密检索索引（FAISS）
  └── bm25/               # BM25 稀疏检索索引（Anserini）
      ├── index/          # Lucene 索引文件
      └── ...
```

---

## 五、常见问题

### Q1: 数据下载速度慢怎么办？

**A**: 可以设置镜像或使用代理：

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### Q2: 如何只使用本地数据，不上传 HuggingFace？

**A**: 可以准备本地数据文件：

```python
# 1. 准备 JSONL 格式的数据
# 2. 使用 datasets 库加载
from datasets import load_dataset
dataset = load_dataset('json', data_files='local_data.jsonl')
```

### Q3: 如何构建自定义索引？

**A**: 参考 `search_r1/search/build_index.sh`：

```bash
# 1. 准备语料库（JSONL 格式）
# 2. 使用 E5 或其他模型编码
# 3. 构建 FAISS 索引
python -c "
import faiss
index = faiss.IndexFlatIP(768)  # 768 是 E5-base 的维度
# 添加向量...
faiss.write_index(index, 'custom.index')
"
```

### Q4: 数据需要多大的存储空间？

**估算**：
- Wiki-18 语料库（压缩）：~30 GB
- Wiki-18 语料库（解压）：~50 GB
- E5 索引（flat）：~70 GB
- 训练数据（Parquet）：每个数据集 ~100 MB

**总计约 150 GB**

### Q5: 如何处理不同的检索器？

**当前支持的检索器**：
- **BM25**: 稀疏检索（`pyserini`）
- **E5**: 稠密检索（`intfloat/e5-base-v2`）
- **BGE**: 稠密检索（`BAAI/bge-base-en-v1.5`）
- **在线搜索**: Google, Bing, Brave Search

**使用方式**：

```bash
# 修改 retrieval_launch.sh 中的参数
--retriever_name e5        # 使用 E5
--retriever_name bge       # 使用 BGE
--retriever_name bm25      # 使用 BM25
```

---

## 六、快速验证数据

### 6.1 检查训练数据

```bash
# 查看训练数据样本
python -c "
import pandas as pd
df = pd.read_parquet('./data/nq_search/train.parquet')
print(df.head())
"
```

### 6.2 检查语料库

```bash
# 查看语料库前几行
head -n 5 /path/to/wiki-18.jsonl
```

### 6.3 检查索引

```python
# 检查 FAISS 索引
import faiss
index = faiss.read_index('/path/to/e5_Flat.index')
print(f"Index dimension: {index.d}, Vector count: {index.ntotal}")
```

---

## 七、参考命令总结

### 最小化启动（使用预计算索引）

```bash
# 1. 下载所有必需文件
python scripts/download.py --save_path ./data
cat ./data/part_* > ./data/e5_Flat.index
gzip -d ./data/wiki-18.jsonl.gz

# 2. 处理数据
python scripts/data_process/nq_search.py --local_dir ./data/nq_search

# 3. 配置检索服务器（retrieval_launch.sh）
# 修改 index_path 和 corpus_path 指向下载的文件

# 4. 启动
bash retrieval_launch.sh
bash train_ppo.sh
```

### 使用自己的数据

```bash
# 1. 准备语料库（JSONL 格式）
cat > my_corpus.jsonl << EOF
{"id": "0", "contents": "\"Title\"\nContent..."}
EOF

# 2. 构建索引
python search_r1/search/index_builder.py \
    --corpus_path my_corpus.jsonl \
    --index_path ./index/my_index \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2

# 3. 更新服务器配置
# 修改 retrieval_launch.sh 指向新索引
```

---

## 总结

- ✅ **训练数据**：从 HuggingFace 自动加载
- ✅ **检索数据**：通过 `scripts/download.py` 下载
- ✅ **语料库**：Wiki-18（约 50 GB）
- ✅ **索引**：E5 索引（约 70 GB）或 BM25 索引
- ✅ **总空间**：约 150 GB

快速开始只需运行：
```bash
python scripts/download.py --save_path ./data
```


