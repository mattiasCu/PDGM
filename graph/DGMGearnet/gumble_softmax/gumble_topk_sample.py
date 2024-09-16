import torch
import numpy as np
import torch.nn as nn
from DGMGearnet.gumble_softmax.deterministic_scheme import select_from_edge_candidates


EPSILON = np.finfo(np.float32).tiny
LARGE_NUMBER = 1.e10

class SubsetOperator(nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON], device=scores.device))
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=2)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=2)
            khot_hard = khot_hard.scatter_(2, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

class GumbleSampler(nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(GumbleSampler, self).__init__()
        self.subset_operator = SubsetOperator(k, tau, hard)
        self.k = k

    def forward(self, x):
        x = self.subset_operator(x)
        return x
    
    def validate(self, x):
        return select_from_edge_candidates(x, self.k)
