"""
张量处理辅助工具模块

本模块提供了用于处理多轮对话中张量操作的辅助功能，包括：
- 张量截断和裁剪
- 填充结构转换
- 注意力掩码和位置ID的创建
- 批量级别的填充处理
"""

import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    """
    张量配置类
    
    用于存储张量处理相关的配置参数
    
    Attributes:
        pad_token_id: 填充标记的token ID
        max_prompt_length: Prompt的最大长度
        max_obs_length: 观察值（如检索结果）的最大长度
        max_start_length: 初始prompt的最大长度
    """
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int

class TensorHelper:
    """
    张量处理辅助类
    
    提供多轮对话中所需的各种张量操作，包括截断、填充、掩码创建等
    """
    def __init__(self, config: TensorConfig):
        """
        初始化张量辅助工具
        
        Args:
            config: 张量配置对象
        """
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                            keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """
        根据注意力掩码将张量裁剪到有效长度
        
        在多轮对话中，为了避免prompt过长导致显存不足，需要裁剪到有效长度。
        有效长度是指所有样本中注意力掩码为1的最大长度。
        
        Args:
            tensor_dict: 包含多个张量的字典
            keys: 需要裁剪的键列表（如 'input_ids', 'attention_mask'）
            cut_left: 是否从左侧裁剪（True表示保留右侧，False表示保留左侧）
        
        Returns:
            裁剪后的张量字典
        """
        # 计算有效长度：找出所有样本中非填充token的最大数量
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()
        
        # 对指定的键进行裁剪
        for key in keys:
            if cut_left:
                # 保留右侧的有效部分
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                # 保留左侧的有效部分
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        转换填充结构并返回排序后的张量及索引
        
        在多轮对话中，需要将padding统一到一侧（左侧或右侧）以便于处理。
        这个方法会重新排列tokens，将非填充token和填充token分组。
        
        Args:
            tensor: 输入的张量
            pad_to_left: 是否将填充移到左侧（True表示移到左侧，False表示移到右侧）
        
        Returns:
            (sorted_tensor, sorted_indices): 排序后的张量和排序索引
        """
        # 创建掩码：标识哪些是填充token
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        # 稳定排序，返回排序后的索引
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        从input_ids创建注意力掩码
        
        注意力掩码用于标识哪些位置是真实token，哪些是填充token。
        1表示真实token，0表示填充token。
        
        Args:
            input_ids: 输入的token ID张量
        
        Returns:
            注意力掩码张量，形状与input_ids相同
        """
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        从注意力掩码创建位置ID
        
        位置ID用于模型了解每个token的位置信息。
        通过累积和来计算位置，填充位置的位置ID为0。
        
        Args:
            attention_mask: 注意力掩码张量
        
        Returns:
            位置ID张量，形状与attention_mask相同
        """
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                               pad_to_left: bool = True) -> torch.Tensor:
        """
        拼接多个张量并处理填充结构
        
        将多个张量在序列维度（dim=1）上拼接，然后统一填充位置。
        
        Args:
            tensors: 需要拼接的张量列表
            pad_to_left: 是否将填充移到左侧
        
        Returns:
            拼接并重排后的张量
        """
        # 在序列维度上拼接
        concatenated = torch.cat(tensors, dim=1)
        # 统一填充结构
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(self, responses: torch.Tensor, 
                          responses_str: List[str], 
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        在样本级别上为不活跃的样本填充pad tokens
        
        在多轮对话中，有些样本可能已经结束（回答了问题），不再需要继续生成。
        为了保持batch的完整性，需要对已经结束的样本用pad tokens填充。
        
        Args:
            responses: 响应张量，只包含活跃样本的响应
            responses_str: 响应字符串列表
            active_mask: 活跃掩码，标识哪些样本还在生成中（True=活跃，False=已结束）
        
        Returns:
            (padded_responses, padded_responses_str): 填充后的响应张量和字符串列表
        """
        # 确保活跃样本数量与响应张量的batch size一致
        assert active_mask.sum() == responses.shape[0]
        
        # 创建全填充的响应张量
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        # 只在活跃位置填充真实响应
        padded_responses[active_mask] = responses
        
        # 创建填充后的响应字符串列表
        padded_responses_str = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                # 只给活跃样本分配响应字符串
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses, padded_responses_str