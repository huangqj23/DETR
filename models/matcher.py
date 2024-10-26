'''
Author: huangqj23
Date: 2024-10-25 11:15:17
LastEditors: huangqj23
LastEditTime: 2024-10-26 16:59:35
FilePath: /DETR/models/matcher.py
Description: 

Copyright (c) 2024 by FutureAI, All Rights Reserved. 
'''

import torch
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    """这个类计算网络的目标和预测之间的分配。
    出于效率原因，目标不包括无对象（no_object）。因此，通常情况下，预测的数量超过目标的数量。
    在这种情况下，我们对最佳预测进行一对一匹配，而其他的则不匹配（因此被视为非对象）。
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        '''Create the matcher
        
        Params: 
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        '''
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ 执行匹配

        参数:
            outputs: 这是一个字典，至少包含以下条目:
                 "pred_logits": 维度为 [batch_size, num_queries, num_classes] 的张量，包含分类 logits
                 "pred_boxes": 维度为 [batch_size, num_queries, 4] 的张量，包含预测框坐标

            targets: 这是一个目标列表 (len(targets) = batch_size)，其中每个目标是一个字典，包含:
                 "labels": 维度为 [num_target_boxes] 的张量 (其中 num_target_boxes 是目标中真实对象的数量)，包含类标签
                 "boxes": 维度为 [num_target_boxes, 4] 的张量，包含目标框坐标

        返回:
            一个大小为 batch_size 的列表，包含 (index_i, index_j) 的元组，其中:
                - index_i 是所选预测的索引 (按顺序)
                - index_j 是相应所选目标的索引 (按顺序)
            对于每个批次元素，它包含:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
