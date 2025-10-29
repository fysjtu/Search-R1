"""
检索模块

本模块提供了文本检索的核心功能，包括：
- 本地稀疏检索器（BM25）
- 本地稠密检索器（Dense Retriever with FAISS）
- 编码器模块（Encoder）
- 语料库加载和文档处理
"""

import json
import os
import warnings
from typing import List, Dict
import functools
from tqdm import tqdm
from multiprocessing import Pool
import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
import argparse
import datasets


def load_corpus(corpus_path: str):
    """
    加载语料库
    
    Args:
        corpus_path: 语料库文件路径（jsonl格式）
    
    Returns:
        语料库数据集对象
    """
    corpus = datasets.load_dataset(
            'json', 
            data_files=corpus_path,
            split="train",
            num_proc=4)
    return corpus
    

def read_jsonl(file_path):
    """
    读取JSONL文件
    
    Args:
        file_path: JSONL文件路径
    
    Returns:
        数据列表
    """
    data = []
    
    with open(file_path, "r") as f:
        readin = f.readlines()
        for line in readin:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    """
    根据索引加载文档
    
    Args:
        corpus: 语料库
        doc_idxs: 文档索引列表
    
    Returns:
        文档列表
    """
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results


def load_model(
        model_path: str, 
        use_fp16: bool = False
    ):
    """
    加载检索模型和分词器
    
    Args:
        model_path: 模型路径（HuggingFace模型）
        use_fp16: 是否使用半精度
    
    Returns:
        (model, tokenizer): 模型和分词器
    """
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask = None,
        pooling_method = "mean"
    ):
    """
    池化操作，将序列表示转换为单一向量
    
    支持三种池化方法：
    - mean: 平均池化
    - cls: 使用[CLS] token表示
    - pooler: 使用pooler输出
    
    Args:
        pooler_output: Pooler输出
        last_hidden_state: 最后隐藏层状态
        attention_mask: 注意力掩码
        pooling_method: 池化方法
    
    Returns:
        池化后的向量表示
    """
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    """
    编码器类
    
    负责将文本（查询或文档）编码为向量表示，用于稠密检索。
    支持多种检索模型：E5、BGE、DPR、T5等。
    """
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        """
        初始化编码器
        
        Args:
            model_name: 模型名称（如 'e5', 'bge'）
            model_path: 模型路径
            pooling_method: 池化方法（'mean', 'cls', 'pooler'）
            max_length: 最大序列长度
            use_fp16: 是否使用半精度
        """
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        # 加载模型和分词器
        self.model, self.tokenizer = load_model(model_path=model_path,
                                                use_fp16=use_fp16)

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        """
        编码文本为向量
        
        根据不同的模型类型，对查询或文档添加不同的前缀：
        - E5: "query: " 或 "passage: "
        - BGE: "Represent this sentence for searching relevant passages: "
        
        Args:
            query_list: 文本列表（查询或文档）
            is_query: 是否为查询（True=查询，False=文档）
        
        Returns:
            编码后的向量数组（numpy array）
        """
        # 处理单个字符串的情况
        if isinstance(query_list, str):
            query_list = [query_list]

        # 为E5模型添加前缀
        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        # 为BGE模型添加前缀
        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        # 分词
        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        # 移动到GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # T5模型需要decoder输入
        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            # 标准BERT-like模型
            output = self.model(**inputs, return_dict=True)
            # 池化操作
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            # DPR不需要归一化，其他模型需要
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        # 转换为numpy数组
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb


class BaseRetriever:
    """
    检索器基类
    
    所有检索器（BM25、Dense等）的抽象基类，定义了统一的接口。
    """

    def __init__(self, config):
        """
        初始化检索器基类
        
        Args:
            config: 配置对象，包含检索方法、topk等参数
        """
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score:bool) -> List[Dict[str, str]]:
        """
        检索文档（单个查询）
        
        子类需要实现此方法来提供具体的检索逻辑。
        
        Args:
            query: 查询字符串
            num: 返回文档数量
            return_score: 是否返回相关性分数
        
        Returns:
            文档列表，每个文档包含contents、title、text等字段
        """
        pass

    def _batch_search(self, query_list, num, return_score):
        """
        批量检索文档
        
        子类需要实现此方法来提供批量检索逻辑。
        
        Args:
            query_list: 查询列表
            num: 返回文档数量
            return_score: 是否返回相关性分数
        
        Returns:
            文档列表的列表
        """
        pass

    def search(self, *args, **kwargs):
        """公开的search接口"""
        return self._search(*args, **kwargs)
    
    def batch_search(self, *args, **kwargs):
        """公开的batch_search接口"""
        return self._batch_search(*args, **kwargs)


class BM25Retriever(BaseRetriever):
    """
    BM25 稀疏检索器
    
    基于预构建的 Pyserini/Anserini BM25 索引进行检索。
    BM25是经典的稀疏检索算法，基于词频和逆文档频率计算相关性。
    
    优点：速度快，不需要GPU，适合大规模语料库
    缺点：无法捕获语义信息，准确率相对较低
    """

    def __init__(self, config):
        """
        初始化BM25检索器
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        # 加载Lucene搜索器
        self.searcher = LuceneSearcher(self.index_path)
        # 检查索引是否包含文档内容
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            # 如果索引不包含文档内容，需要加载语料库
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
        
    def _check_contain_doc(self):
        """
        检查索引是否包含文档内容
        
        如果索引包含原始文档内容，则可以直接从索引中获取；
        否则需要从语料库文件中加载。
        
        Returns:
            是否包含文档内容
        """
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score = False) -> List[Dict[str, str]]:
        """
        BM25单个查询检索
        
        使用Lucene索引进行BM25检索，返回相关文档。
        
        Args:
            query: 查询字符串
            num: 返回文档数量
            return_score: 是否返回BM25分数
        
        Returns:
            文档列表（或文档列表和分数）
        """
        if num is None:
            num = self.topk
        
        # 执行BM25搜索
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [],[]
            else:
                return []
            
        # 提取分数
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        # 从索引或语料库加载文档内容
        if self.contain_doc:
            # 从索引中直接获取文档内容
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in hits]
            results = [{'title': content.split("\n")[0].strip("\""), 
                        'text': "\n".join(content.split("\n")[1:]),
                        'contents': content} for content in all_contents]
        else:
            # 从语料库文件加载文档
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list, num: int = None, return_score = False):
        """
        BM25批量检索
        
        对多个查询逐个执行搜索（未优化，TODO: 可以使用并行）
        
        Args:
            query_list: 查询列表
            num: 返回文档数量
            return_score: 是否返回分数
        
        Returns:
            文档列表的列表（或包含分数）
        """
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num,True)
            results.append(item_result)
            scores.append(item_score)

        if return_score:
            return results, scores
        else:
            return results

def get_available_gpu_memory():
    memory_info = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        free_memory = total_memory - allocated_memory
        memory_info.append((i, free_memory / 1e9))  # Convert to GB
    return memory_info


class DenseRetriever(BaseRetriever):
    """
    稠密检索器
    
    基于预构建的FAISS索引进行向量检索。
    使用深度学习的embedding模型编码查询和文档，通过向量相似度计算相关性。
    
    优点：能够捕获语义信息，准确率高
    缺点：需要GPU，索引构建耗时
    
    支持的类型：
    - Flat: 暴力搜索（最准确，但较慢）
    - IVF: 倒排索引（速度快，准确率稍低）
    - HNSW: 层级导航小世界图（速度快且准确）
    """

    def __init__(self, config: dict):
        """
        初始化稠密检索器
        
        Args:
            config: 配置对象，包含索引路径、模型路径等
        """
        super().__init__(config)
        # 加载FAISS索引
        self.index = faiss.read_index(self.index_path)
        # 如果启用GPU，将索引移到GPU
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True  # 使用半精度节省显存
            co.shard = True  # 跨GPU分片
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        # 加载语料库
        self.corpus = load_corpus(self.corpus_path)
        # 初始化编码器
        self.encoder = Encoder(
             model_name = self.retrieval_method, 
             model_path = config.retrieval_model_path,
             pooling_method = config.retrieval_pooling_method,
             max_length = config.retrieval_query_max_length,
             use_fp16 = config.retrieval_use_fp16
            )
        self.topk = config.retrieval_topk
        self.batch_size = self.config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score = False):
        """
        稠密检索单个查询
        
        1. 编码查询为向量
        2. 在FAISS索引中搜索最相似的文档
        3. 返回文档列表
        
        Args:
            query: 查询字符串
            num: 返回文档数量
            return_score: 是否返回相似度分数
        
        Returns:
            文档列表（或文档列表和分数）
        """
        if num is None:
            num = self.topk
        # 编码查询
        query_emb = self.encoder.encode(query)
        # 在FAISS索引中搜索
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]

        # 加载文档内容
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score = False):
        """
        稠密检索批量查询（优化版本）
        
        使用批量处理提高效率：
        1. 批量编码查询
        2. 批量在FAISS中搜索
        3. 批量加载文档
        
        Args:
            query_list: 查询字符串列表
            num: 每个查询返回的文档数量
            return_score: 是否返回分数
        
        Returns:
            文档列表的列表（每个查询对应一个文档列表）
        """
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        batch_size = self.batch_size

        results = []
        scores = []

        # 分批处理查询
        for start_idx in tqdm(range(0, len(query_list), batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + batch_size]
            
            # 批量编码
            batch_emb = self.encoder.encode(query_batch)
            # 批量搜索
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            
            # 加载文档
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # 重新分组：每个查询对应num个文档
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            scores.extend(batch_scores)
            results.extend(batch_results)
        
        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    """
    根据配置自动选择检索器类型
    
    根据retrieval_method配置选择相应的检索器：
    - "bm25": 返回BM25Retriever（稀疏检索）
    - 其他: 返回DenseRetriever（稠密检索）
    
    Args:
        config: 配置对象，必须包含retrieval_method字段
    
    Returns:
        检索器实例（BM25Retriever或DenseRetriever）
    """
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


def get_dataset(config):
    """
    从配置加载数据集
    
    Args:
        config: 配置对象，包含dataset_path和data_split
    
    Returns:
        数据集对象
    """
    split_path = os.path.join(config.dataset_path, f'{config.data_split}.jsonl')
    return read_jsonl(split_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "Retrieval")

    # Basic parameters
    parser.add_argument('--retrieval_method', type=str)
    parser.add_argument('--retrieval_topk', type=int, default=10)
    parser.add_argument('--index_path', type=str, default=None)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--dataset_path', default=None, type=str)

    parser.add_argument('--faiss_gpu', default=True, type=bool)
    parser.add_argument('--data_split', default="train", type=str)
    
    parser.add_argument('--retrieval_model_path', type=str, default=None)
    parser.add_argument('--retrieval_pooling_method', default='mean', type=str)
    parser.add_argument('--retrieval_query_max_length', default=256, type=str)
    parser.add_argument('--retrieval_use_fp16', action='store_true', default=False)
    parser.add_argument('--retrieval_batch_size', default=512, type=int)
    
    args = parser.parse_args()

    args.index_path = os.path.join(args.index_path, f'{args.retrieval_method}_Flat.index') if args.retrieval_method != 'bm25' else os.path.join(args.index_path, 'bm25')

    # load dataset
    all_split = get_dataset(args)
    
    input_query = [sample['question'] for sample in all_split[:512]]
    
    # initialize the retriever and conduct retrieval
    retriever = get_retriever(args)
    print('Start Retrieving ...')    
    results, scores = retriever.batch_search(input_query, return_score=True)

    # from IPython import embed
    # embed()
