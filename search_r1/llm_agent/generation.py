"""
LLM生成管理模块

本模块实现了多轮对话中LLM与搜索引擎的协作生成逻辑，核心功能包括：
- 多轮生成循环管理
- 搜索动作解析和执行
- 状态管理和信息屏蔽
- 批处理优化
"""

import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    """
    生成配置类
    
    用于配置多轮对话生成的各种参数
    
    Attributes:
        max_turns: 最大对话轮数（包括搜索和最终回答）
        max_start_length: 初始prompt的最大长度
        max_prompt_length: 整个prompt的最大长度（用于截断）
        max_response_length: 单轮响应的最大长度
        max_obs_length: 观察值（如检索结果）的最大长度
        num_gpus: GPU数量（用于批处理优化）
        no_think_rl: 是否在RL训练中屏蔽推理部分
        search_url: 搜索引擎的URL地址
        topk: 每次检索返回的文档数量
    """
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    """
    LLM生成管理器
    
    负责管理多轮对话中的LLM生成、搜索动作执行、状态更新等核心逻辑。
    支持LLM与搜索引擎的协作推理。
    """
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        """
        初始化LLM生成管理器
        
        Args:
            tokenizer: 分词器
            actor_rollout_wg: Actor-Rollout工作组件，用于生成序列
            config: 生成配置对象
            is_validation: 是否为验证模式（验证模式下可能跳过搜索）
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        # 初始化张量辅助工具
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """
        批量将响应文本转换为token IDs
        
        Args:
            responses: 响应字符串列表
        
        Returns:
            转换后的token ID张量
        """
        return self.tokenizer(
            responses, 
            add_special_tokens=False,  # 不添加特殊token
            return_tensors='pt', 
            padding="longest"  # 自动填充到batch中最长序列的长度
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """
        后处理响应，在搜索操作或答案操作处停止
        
        这个方法用于清理模型生成的响应，只在完整的动作标签处停止。
        例如：如果生成了 `<search>query</search>some text`，会截断为 `<search>query</search>`
        
        Args:
            responses: 原始响应张量
        
        Returns:
            (processed_responses, responses_str): 处理后的响应张量和字符串列表
        """
        # 解码为字符串
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # 在搜索或答案标签处停止
        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        # 如果启用no_think_rl模式（目前未使用）
        if self.config.no_think_rl:
            raise ValueError('stop')
            # 只保留动作标签，移除推理内容
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        
        # 重新tokenize
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """
        处理来自环境的下一轮观察值
        
        观察值通常是检索到的信息，需要转换为token IDs并检查长度。
        如果观察值过长，会被截断。
        
        Args:
            next_obs: 观察值字符串列表（通常是检索结果）
        
        Returns:
            转换后的token ID张量
        """
        # 将观察值转换为token IDs
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # 不添加特殊token
        )['input_ids']

        # 如果观察值过长，截断到最大长度
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """
        更新滚动状态，将新的响应和观察值加入历史
        
        这个方法用于在多轮对话中维护状态，将：
        - 历史prompt + 当前响应 + 检索结果 拼接在一起
        
        Args:
            rollings: 当前的历史状态（DataProto对象）
            cur_responses: 当前轮次生成的响应
            next_obs_ids: 下一轮观察值（通常是检索结果）
        
        Returns:
            更新后的新状态（DataProto对象）
        """
        # 拼接：历史prompt + 当前响应 + 观察值，并处理padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # 创建注意力掩码和位置ID
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # 截断到合适的长度（避免过长）
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        # 创建新的状态
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],  # 只保留右侧的有效部分
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        # 保留元信息
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """
        拼接张量并处理padding，同时创建信息掩码
        
        这是Search-R1的核心创新之一：信息屏蔽机制。
        创建两个版本：
        1. padded_tensor: 完整版本（包含检索信息，用于提供上下文）
        2. padded_tensor_with_info: 屏蔽版本（检索部分用pad_token_id替换，用于奖励计算）
        
        目的：在RL训练时，奖励只针对LLM的推理部分，不包括检索到的信息。
        这样避免模型过度依赖检索质量，只学习如何更好地利用检索信息。
        
        Args:
            prompt: Prompt张量
            prompt_with_mask: Prompt的掩码版本（初始时通常与prompt相同）
            response: 响应张量
            info: 信息张量（通常是检索结果），可选
            pad_to_left: 是否将padding移到左侧
        
        Returns:
            (padded_tensor, padded_tensor_with_info): 完整版本和屏蔽版本
        """
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        
        # 如果存在检索信息
        if info is not None:
            tensors.append(info)
            # 创建全为pad_token_id的信息掩码（用于屏蔽检索部分）
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)
        
        # 拼接完整版本
        concatenated = torch.cat(tensors, dim=1)
        # 拼接掩码版本
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        
        # 统一padding结构
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        运行主LLM生成循环
        
        这是Search-R1的核心方法，实现多轮协作推理：
        1. 初始化状态
        2. 循环最多max_turns轮：
           - LLM生成响应
           - 解析动作（search/answer）
           - 执行动作（调用搜索引擎或生成答案）
           - 更新状态
        3. 最终生成答案（如果有未完成的样本）
        
        返回包含prompt、responses和info_mask的完整数据，用于RL训练。
        
        Args:
            gen_batch: 初始生成批次
            initial_input_ids: 初始输入ID（用户问题）
        
        Returns:
            包含完整生成数据的DataProto对象
        """
        # 初始化左侧（初始prompt）和右侧（所有生成的responses）
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        # 活跃掩码：标记哪些样本还在生成中（未到达答案）
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        # 统计信息
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)  # 每轮都递增
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)  # 有效动作数
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)  # 搜索次数
        active_num_list = [active_mask.sum().item()]  # 活跃样本数历史
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        执行预测动作，处理搜索和答案操作
        
        这是多轮对话的环境step函数，负责：
        1. 解析LLM生成的响应，提取动作类型（search/answer）
        2. 如果是搜索动作，批量调用搜索引擎
        3. 如果是答案动作，标记为完成
        4. 返回观察值（检索结果或空）
        
        Args:
            predictions: LLM生成的响应列表
            pad_token: 填充token（用于不活跃样本）
            active_mask: 活跃样本掩码（哪些样本还在生成中）
            do_search: 是否执行搜索（在最终轮可能为False）
            
        Returns:
            (next_obs, dones, valid_action, is_search):
            - next_obs: 下一轮观察值（检索结果或空字符串）
            - dones: 是否完成的布尔列表
            - valid_action: 是否有效动作的列表
            - is_search: 是否为搜索动作的列表
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        后处理LLM预测，提取动作类型和内容
        
        解析LLM生成的文本中的动作标签：
        - <search>query</search>: 搜索动作
        - <answer>content</answer>: 答案动作
        
        Args:
            predictions: 原始预测列表（LLM生成的文本）
            
        Returns:
            (actions, contents):
            - actions: 动作类型列表（'search'、'answer'或None）
            - contents: 动作内容列表（查询字符串或答案内容）
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        批量搜索，处理多个查询
        
        将多个搜索查询批量发送到搜索引擎，并将结果格式化。
        这是批处理优化的关键，避免多次网络调用。
        
        Args:
            queries: 查询字符串列表
        
        Returns:
            格式化后的检索结果字符串列表
        """
        # 调用HTTP API执行批量搜索
        results = self._batch_search(queries)['result']
        
        # 将检索结果格式化为字符串
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        """
        内部批量搜索方法，发送HTTP请求到检索服务器
        
        Args:
            queries: 查询字符串列表
        
        Returns:
            包含检索结果的JSON响应
        """
        payload = {
            "queries": queries,
            "topk": self.config.topk,  # 返回top-k文档
            "return_scores": True  # 返回相关性分数
        }
        
        # 发送POST请求到检索服务器
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        """
        将检索结果格式化为字符串
        
        将检索到的文档格式化为便于LLM理解的形式：
        Doc 1(Title: ...) content...
        Doc 2(Title: ...) content...
        
        Args:
            retrieval_result: 检索结果列表（每个元素包含document信息）
        
        Returns:
            格式化后的字符串
        """
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            # 提取文档内容
            content = doc_item['document']['contents']
            # 第一行是标题
            title = content.split("\n")[0]
            # 其余部分是文本
            text = "\n".join(content.split("\n")[1:])
            # 格式化输出
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
