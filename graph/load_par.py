from torchdrug import  layers, datasets,transforms,core, models
from torchdrug.core import Registry as R
from torchdrug.layers import geometry

import torch
from torchdrug import data

from torch_scatter import scatter_add
import torch.nn as nn
from torchdrug import utils
from torch.utils import checkpoint
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat, pack, unpack
import matplotlib.pyplot as plt

from DGMGearnet.model import DGMGearnet_only_sequence, DGMGearnet_edge
from graph_rewiring import GraphRewiring

import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import networkx as nx
import numpy


def adjacency_matrix_to_edge_list(A):
    # 找到非零元素的索引
    idx = torch.nonzero(A, as_tuple=False) 
    i = idx[:, 0]  
    j = idx[:, 1]
    line_num = A.size(0)  
    
    # 计算关系编号（relation）和调整后的列索引（j_prime）
    relation = torch.div(j, line_num, rounding_mode='floor')  # 关系编号，0到6
    j_prime = j % line_num  # 调整后的列索引，0到184
    
    # 组合得到edge_list矩阵，形状为[n, 3]
    edge_list = torch.stack([i, j_prime, relation], dim=1)
    
    # 根据edge_list的索引顺序，获取对应的edge_weighted矩阵
    edge_weighted = A[i, j]
    
    return edge_list, edge_weighted


EnzymeCommission = R.search("datasets.EnzymeCommission")
PV = R.search("transforms.ProteinView")
trans = PV(view = "residue")
dataset = EnzymeCommission("~/scratch/protein-datasets/", test_cutoff=0.95, 
                           atom_feature="full", bond_feature="full", verbose=1, transform = trans)

# 只保留alpha碳的简化格式
graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)
                                                                 ],
                                                    edge_feature="gearnet"
                                                    )

graphs = dataset[0:1]
graphs = [element["graph"] for element in graphs]
graphs = data.Protein.pack(graphs)
graph = graph_construction_model(graphs)


relation_dims = [[21, 128, 512, 512], [4096, 1024, 512, 512]]
score_in_dim = 512
score_out_dim = 512
diffusion_dims = [[21, 512, 512, 512], [512, 512, 512, 512]]  
num_relations = graph.num_relation
attn_num_relation = 2
num_heads = 8
window_size = 10
k = 5
edge_input_dim = 59
num_angle_bin = 8

model = DGMGearnet_edge(relation_dims, score_in_dim, score_out_dim, diffusion_dims, num_relations, attn_num_relation, num_heads, window_size, k, 
                        edge_input_dim, num_angle_bin, batch_norm=True, sample = False,activation="relu", concat_hidden=True, edge_feature="gearnet", readout="sum")

output = model(graph.to(device), graph.node_feature.float().to(device))
graph_feature = output["graph_feature"]





"""
# 加载模型参数并画图
model_path = 'DGMGearnet/model_epoch_92.pth'
checkpoint = torch.load(model_path)

# 如果checkpoint包含多个键，提取出模型的state_dict
if 'model' in checkpoint:
    model_state_dict = checkpoint['model']
else:
    model_state_dict = checkpoint

# 加载模型参数，使用strict=False来忽略不匹配的键
model.load_state_dict(model_state_dict, strict=False)


with torch.no_grad():
    output = model.to(device)(graph.to(device), graph.node_feature.to(device).float())
    
edge_list = output["edge_list"]


# 将张量拆分成7个600x600的子张量
num_subgraphs = 7
subgraph_size = 185

# 拆分张量
subgraphs = torch.split(edge_list, subgraph_size, dim=1)

# 绘制每个子图
for i, subgraph in enumerate(subgraphs):
    # 创建网络图
    G = nx.Graph()
    
    # 遍历子张量中的元素，添加节点和边
    for j in range(subgraph_size):
        for k in range(subgraph_size):
            if subgraph[j, k] != 0:  # 假设非零值表示存在边
                G.add_edge(j, k, weight=subgraph[j, k].item())
    
    # 绘制图形
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_size=70, node_color='lightblue', font_size=6, font_color='black', edge_color='black')
    plt.title(f'Graph {i + 1}')
    plt.savefig(f'fig/graph_{i + 1}.png')


print(output)
"""