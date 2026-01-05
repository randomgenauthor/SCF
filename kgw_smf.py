# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# kgw_smf.py
# Description: Implementation of KGW algorithm
# ============================================

import torch
import os
from math import sqrt
from functools import partial
from ..base import BaseWatermark, BaseConfig
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class KGW_SMFConfig(BaseConfig):
    """Config class for KGW_SMF algorithm"""
    
    # 初始化KGW_SMF特有的参数
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.gamma = self.config_dict['gamma']
        self.delta = self.config_dict['delta']
        self.hash_key = self.config_dict['hash_key']
        self.z_threshold = self.config_dict['z_threshold']
        self.prefix_length = self.config_dict['prefix_length']
        self.f_scheme = self.config_dict['f_scheme']
        self.window_scheme = self.config_dict['window_scheme']
        # SMF 相关：mode=none/hf/ssq；smf_head_k用于hf头部大小
        self.smf_mode = self.config_dict.get('smf_mode', 'none')
        self.smf_head_k = self.config_dict.get('smf_head_k', 0)
        # hf检测时是否重放头部重分配（1开启，0则沿用KGW检测）
        self.smf_hf_detect = self.config_dict.get('smf_hf_detect', 0)
        # ssq: semantic cluster ids path (.pt tensor of shape [vocab_size])
        self.smf_cluster_path = self.config_dict.get('smf_cluster_path', None)
    
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'KGW_SMF'

class KGW_SMFUtils:
    """Utility class for KGW_SMF algorithm, contains helper functions."""

    # 工具类初始化：配置随机生成器和PRF
    def __init__(self, config: KGW_SMFConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW_SMF utility class.

            Parameters:
                config (KGW_SMFConfig): Configuration for the KGW_SMF algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)
        self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng)
        self.f_scheme_map = {"time": self._f_time, "additive": self._f_additive, "skip": self._f_skip, "min": self._f_min}
        self.window_scheme_map = {"left": self._get_greenlist_ids_left, "self": self._get_greenlist_ids_self}

        self._ssq_cluster_token_ids: dict[int, torch.LongTensor] | None = None
        if getattr(self.config, "smf_mode", "none") == "ssq":
            self._init_ssq_clusters()

    def _init_ssq_clusters(self) -> None:
        cluster_path = getattr(self.config, "smf_cluster_path", None)
        if not cluster_path:
            raise ValueError("smf_mode='ssq' requires 'smf_cluster_path' in config.")

        if not os.path.exists(cluster_path):
            raise FileNotFoundError(f"SSQ cluster file not found: {cluster_path}")

        cluster_ids = torch.load(cluster_path, map_location="cpu")
        if not isinstance(cluster_ids, torch.Tensor):
            raise TypeError(f"SSQ cluster file must contain a torch.Tensor, got {type(cluster_ids)}")
        if cluster_ids.numel() != int(self.config.vocab_size):
            raise ValueError(
                f"SSQ cluster ids length {cluster_ids.numel()} != vocab_size {self.config.vocab_size}. "
                f"Make sure clusters were built for the same tokenizer/model."
            )
        if cluster_ids.dtype != torch.long:
            cluster_ids = cluster_ids.long()

        cluster_to_tokens: dict[int, list[int]] = {}
        for token_id, cluster_id in enumerate(cluster_ids.tolist()):
            cluster_to_tokens.setdefault(int(cluster_id), []).append(token_id)

        self._ssq_cluster_token_ids = {
            cid: torch.tensor(tids, dtype=torch.long, device=self.config.device)
            for cid, tids in cluster_to_tokens.items()
            if len(tids) > 0
        }

    def get_ssq_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        SSQ: sample ~gamma proportion within each semantic cluster, then union to build global greenlist.

        Deterministic given the prefix (mirrors KGW leftHash seeding).
        """
        if self._ssq_cluster_token_ids is None:
            self._init_ssq_clusters()

        if self.config.window_scheme != "left":
            raise NotImplementedError("SSQ mode currently supports window_scheme='left' only.")

        base_seed = int((self.config.hash_key * self._f(input_ids)) % max(1, self.config.vocab_size))
        selected_chunks: list[torch.LongTensor] = []

        for cluster_id in sorted(self._ssq_cluster_token_ids.keys()):
            token_ids = self._ssq_cluster_token_ids[cluster_id]
            cluster_size = int(token_ids.numel())
            if cluster_size <= 0:
                continue

            desired = cluster_size * float(self.config.gamma)
            m = int(desired)
            residual = desired - m
            if residual > 0:
                # deterministic stochastic rounding per (prefix, cluster)
                self.rng.manual_seed(base_seed + int(cluster_id))
                if torch.rand((), device=token_ids.device, generator=self.rng).item() < residual:
                    m += 1
            if m <= 0:
                continue

            self.rng.manual_seed(base_seed + int(cluster_id))
            perm = torch.randperm(cluster_size, device=token_ids.device, generator=self.rng)
            selected_chunks.append(token_ids[perm[:m]])

        if not selected_chunks:
            return torch.empty((0,), dtype=torch.long, device=self.config.device)

        return torch.cat(selected_chunks, dim=0)

    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token."""
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))
    
    # 时间乘积哈希
    def _f_time(self, input_ids: torch.LongTensor):
        """Get the previous token time."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        return self.prf[time_result % self.config.vocab_size]
    
    # 累加哈希
    def _f_additive(self, input_ids: torch.LongTensor):
        """Get the previous token additive."""
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        return self.prf[additive_result % self.config.vocab_size]
    
    # 跳跃哈希
    def _f_skip(self, input_ids: torch.LongTensor):
        """Get the previous token skip."""
        return self.prf[input_ids[- self.config.prefix_length].item()]

    # 取前缀最小哈希
    def _f_min(self, input_ids: torch.LongTensor):
        """Get the previous token min."""
        return min(self.prf[input_ids[-1 - i].item()] for i in range(0, self.config.prefix_length))
    
    # 计算当前前缀的绿色列表
    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        return self.window_scheme_map[self.config.window_scheme](input_ids)

    # hf模式下调整绿色列表：当头部绿占比 < gamma/2 时重分配
    def get_hf_adjusted_greenlist_ids(self, prefix_ids: torch.LongTensor, probs: torch.Tensor) -> torch.LongTensor:
        """Return adjusted greenlist under hf mode with head rebalancing."""
        base_green = self.get_greenlist_ids(prefix_ids)
        base_green_t = torch.as_tensor(base_green, device=probs.device, dtype=torch.long)

        k = int(self.config.smf_head_k or 0)
        if k <= 0:
            return base_green_t

        # 取top-k头部
        head_idx = torch.topk(probs, k).indices

        # 计算头部的绿色比例（全张量，避免Python集合开销）
        head_green_count = torch.isin(head_idx, base_green_t).sum().item()
        ratio = head_green_count / max(1, k)

        # 满足比例则不调整
        if ratio >= (self.config.gamma / 2):
            return base_green_t

        # 否则重新分配头部绿色，目标比例约为gamma
        m = int(round(self.config.gamma * k))
        m = max(0, min(m, k))

        # 复用内部rng，前缀相关种子保证确定性
        self.rng.manual_seed((self.config.hash_key * self._f(prefix_ids)) % max(1, self.config.vocab_size))
        perm = torch.randperm(k, device=probs.device, generator=self.rng)
        new_head_green = head_idx[perm[:m]]

        # 去掉原base里的头部token后再并上新的头部绿色
        keep_mask = ~torch.isin(base_green_t, head_idx)
        base_keep = base_green_t[keep_mask]
        adjusted = torch.cat([base_keep, new_head_green], dim=0)
        return torch.unique(adjusted)
    
    # 左窗口方案生成绿色列表
    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        self.rng.manual_seed((self.config.hash_key * self._f(input_ids)) % self.config.vocab_size)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids
    
    # selfHash 方案生成绿色列表
    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via selfHash scheme."""
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        greenlist_ids = []
        f_x = self._f(input_ids)
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf[k])
            self.rng.manual_seed(h_k % self.config.vocab_size)
            vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)
        return greenlist_ids
    
    # 根据命中数计算z分数
    def _compute_z_score(self, observed_count: int , T: int) -> float: 
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z
    
    # 对序列进行打分，返回z分数与绿色标记
    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]

        mode = getattr(self.config, 'smf_mode', 'none')
        hf_detect = bool(getattr(self.config, 'smf_hf_detect', 0))
        model = getattr(self.config, 'generation_model', None)

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            prefix_ids = input_ids[:idx]

            # 原始KGW或未启用hf/ssq
            if mode == 'none':
                greenlist_ids = self.get_greenlist_ids(prefix_ids)
            elif mode == 'hf':
                if hf_detect:
                    if model is None:
                        raise RuntimeError("hf检测需要generation_model以获取logits")
                    with torch.no_grad():
                        logits = model(input_ids=prefix_ids.unsqueeze(0).to(self.config.device)).logits[0, -1, :]
                        probs = torch.softmax(logits, dim=-1)
                    greenlist_ids = self.get_hf_adjusted_greenlist_ids(prefix_ids.to(self.config.device), probs)
                else:
                    greenlist_ids = self.get_greenlist_ids(prefix_ids)
            else:  # ssq留出扩展位，当前沿用KGW greenlist
                greenlist_ids = self.get_ssq_greenlist_ids(prefix_ids)

            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class KGW_SMFLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW_SMF algorithm, process logits to add watermark."""

    # 初始化logits处理器
    def __init__(self, config: KGW_SMFConfig, utils: KGW_SMFUtils, *args, **kwargs) -> None:
        """
            Initialize the KGW_SMF logits processor.

            Parameters:
                config (KGW_SMFConfig): Configuration for the KGW_SMF algorithm.
                utils (KGW_SMFUtils): Utility class for the KGW_SMF algorithm.
        """
        self.config = config
        self.utils = utils

    # 计算绿色token的掩码
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    # 给绿色token增加偏置
    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    # 调用时对logits添加水印偏置
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        # 序列长度不足前缀要求时直接返回原始scores，不加水印
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        mode = getattr(self.config, 'smf_mode', 'none')

        # 预分配每个batch位置的绿色列表
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        # 需要概率时先计算softmax
        if mode == 'hf':
            scores_batch = scores if scores.dim() == 2 else scores.unsqueeze(0)
            probs_batch = torch.softmax(scores_batch, dim=-1)
        else:
            probs_batch = None

        # 为每个batch样本根据前缀计算绿色token集合
        for b_idx in range(input_ids.shape[0]):
            if mode == 'hf':
                greenlist_ids = self.utils.get_hf_adjusted_greenlist_ids(input_ids[b_idx], probs_batch[b_idx])
            elif mode == 'none':
                greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            else:  # ssq 预留扩展位，当前仍使用base green
                greenlist_ids = self.utils.get_ssq_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        # 将绿色token集合转换成布尔掩码
        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        # 对绿色token对应的logits加上偏置delta
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        # 返回偏置后的scores
        return scores
    

class KGW_SMF(BaseWatermark):
    """Top-level class for KGW_SMF algorithm."""

    # 算法初始化，加载配置并构建工具类
    def __init__(self, algorithm_config: str | KGW_SMFConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the KGW_SMF algorithm.

            Parameters:
                algorithm_config (str | KGW_SMFConfig): Path to the algorithm configuration file or KGW_SMFConfig instance.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = KGW_SMFConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, KGW_SMFConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a KGW_SMFConfig instance")
            
        self.utils = KGW_SMFUtils(self.config)
        self.logits_processor = KGW_SMFLogitsProcessor(self.config, self.utils)
    
    # 生成带水印文本
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text
    
    # 检测文本中的水印
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # Encode the text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute z_score using a utility method（hf/ssq检测当前沿用base green，保持简单）
        z_score, _ = self.utils.score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
        
    # 获取可视化所需的数据
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""
        
        # Encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Compute z-score and highlight values
        z_score, highlight_values = self.utils.score_sequence(encoded_text)
        
        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values)
