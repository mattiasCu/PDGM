import torch
import numpy as np

LARGE_NUMBER = 1.e10

def select_from_edge_candidates(scores: torch.Tensor, k: int):
    data = scores
    labels = torch.zeros_like(data)
    _, indices = torch.topk(data, k, dim=2)
    labels.scatter_(2, indices, 1.0)  # 设置top-k位置为1
    return labels
