from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn. functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from utils.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

# 此模块实现了一种特殊的批量归一化（Batch Normalization），其中批量统计数据（均值和方差）以及仿射参数（权重和偏置）是固定的，
# 不会在训练过程中更新。这种实现通常用于某些预训练模型的微调或推理阶段。
# 类的设计目的是在推理阶段使用固定的统计数据和参数进行批量归一化，这在微调预训练模型时可能非常有用，因为它可以保持模型的稳定性和一致性。
class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d，其中批量统计和仿射参数是固定的。 从 torchvision.misc.ops 复制粘贴，
    并在 rqsrt 之前添加了 eps， 否则除了 torchvision.models.resnet[18,34,50,101] 之外的任何其他模型都会产生 nans。
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # 使用 register_buffer 方法注册了四个缓冲区：weight、bias、running_mean 和 running_var。
        # 这些缓冲区不会被优化器更新，因为它们是缓冲区而非参数。
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 写了 _load_from_state_dict 方法以删除状态字典中的 num_batches_tracked 键（如果存在），这是因为在冻结的批量归一化中不需要跟踪批次数。
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        # 调用父类的 _load_from_state_dict 方法继续加载其他状态。
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        
    def forward(self, x):
        # move reshapes to the beginning to make it fuser-friendly
        # 便于与输入张量 x 进行广播运算。
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5 # 用于数值稳定性，避免除以零。
        scale = w * (rv + eps).rsqrt() # 归一化的缩放因子
        bias = b - rm * scale # 归一化的偏置项
        return x * scale + bias # 对输入进行缩放和偏移。
    
class BackboneBase(nn.Module):
    def __init__(self, backbone:nn.Module, train_backbone: bool, num_channels:int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        else:
            return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
    
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
                                                     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        
class Joiner(nn.Sequential): # Joiner 继承自 nn.Sequential，这意味着它是一个顺序的模块容器，能够按顺序执行其中的子模块。
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding) # 调用父类的初始化方法，将 backbone 和 position_embedding 作为 nn.Sequential 的模块序列进行初始化。
        
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list) # 使用 self[0]，即 backbone 模块处理输入的 tensor_list。xs 是一个字典，包含了处理后的特征图。
        out: List[NestedTensor] = [] # 初始化两个列表：out 用于存储处理后的 NestedTensor，pos 用于存储位置编码。
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype)) # 对每个 x，使用 self[1]（即 position_embedding 模块）计算位置编码，并转换为与 x.tensors 相同的数据类型，将结果添加到 pos 列表。
        return out, pos
    

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model