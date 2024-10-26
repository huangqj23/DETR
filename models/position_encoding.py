'''
Author: huangqj23
Date: 2024-10-26 14:53:12
LastEditors: huangqj23
LastEditTime: 2024-10-26 17:35:12
FilePath: /DETR/models/position_encoding.py
Description: 

Copyright (c) 2024 by FutureAI, All Rights Reserved. 
'''
import math
import torch
import torch.nn as nn 
from utils.misc import NestedTensor

# 正弦位置编码是一种固定的、非学习的编码方式，最初在 Transformer 模型中引入，用于在序列模型中加入位置信息。
# 通过使用正弦和余弦函数，模型能够轻松推断出序列中元素的相对位置。
class PositionEmbeddingSine(nn.Module):
    """这是一个更标准的位置信息嵌入版本，与《Attention is all you need》论文中使用的非常相似，已推广到图像处理。"""
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  
        return pos
    
# nn.Embedding 是 PyTorch 中用于创建嵌入层的模块。嵌入层是一种查找表，用于将离散的索引映射到连续的向量表示中。
# 这种映射对于处理序列数据（如文本、图像等）非常有用，因为它能够将离散的输入（如单词或像素位置）转换为可用于神经网络的连续表示。
# 50: 这是嵌入层的输入维度大小，也就是可以嵌入的不同索引的数量。在这个例子中，嵌入层可以处理的最大索引是 49（从 0 开始计数）。这通常表示可以嵌入的最大行数。
# num_pos_feats: 这是每个嵌入向量的维度大小。每个输入索引将被映射到一个具有 num_pos_feats 维度的向量。
class PositionEmbeddingLearned(nn.Module):
    """绝对位置嵌入，已学习。"""
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
    
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i) # 通过 col_embed 将列索引 i 映射到嵌入空间得到的嵌入向量。
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1), # 将 x_emb 的第一个维度扩展并重复 h 次，生成一个形状为 (h, w, num_pos_feats) 的张量，其中每一行都包含相同的列嵌入。
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # 将列嵌入和行嵌入在最后一个维度上拼接，得到一个形状为 (h, w, 2 * num_pos_feats) 的张量。
        # 调整张量的维度顺序，使其变为 (2 * num_pos_feats, h, w)。在最前面增加一个批次维度，并根据输入张量的批次大小重复，最终得到一个形状为 (batch_size, 2 * num_pos_feats, h, w) 的位置编码张量。
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f'not supported {args.position_embedding}.')
    return position_embedding