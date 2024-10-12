from torch_scatter import  scatter_add
import torch.nn as nn
from torchdrug import utils, layers, core, data
from torchdrug.layers import functional
from torch.utils import checkpoint
import torch 

from torchdrug.core import Registry as R
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack


from DGMGearnet.gumble_softmax.gumble_topk_sample import GumbleSampler
from collections.abc import Sequence


LARGE_NUMBER = 1.e10

class relationalGraphConv(layers.MessagePassingBase):
    
    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(relationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None


    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation
        device = input.device  # Ensure device consistency
        
        node_in, node_out, relation = graph.edge_list.t().to(device)
        node_out = node_out * self.num_relation + relation
    
        degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
        edge_weight = graph.edge_weight / degree_out[node_out]
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t().to(device), input.to(device))
        
        if self.edge_linear:
            edge_input = graph.edge_feature.float().to(device)
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1).to(device)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update.view(input.size(0), self.num_relation * self.input_dim)                           


    def combine(self, input, update):
        device = input.device
        self.linear.to(device)  # Ensure the linear layers are on the correct device
        self.self_loop.to(device)
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            self.batch_norm.to(device)
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def forward(self, graph, input):
        device = input.device
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph.to(device), input)
        output = self.combine(input, update)
        return output


# 可rewire的关系图神经网络
@R.register("layer.relationalGraph")
class relationalGraph(layers.MessagePassingBase):
    
    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(relationalGraph, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None
        
    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation
        device = input.device  # Ensure device consistency
        

        node_in, node_out, relation = graph.edge_list.t().to(device)
        node_out = node_out * self.num_relation + relation
    
        degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
        edge_weight = graph.edge_weight / degree_out[node_out]
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t().to(device), input.to(device))
        
        if self.edge_linear:
            edge_input = graph.edge_feature.float().to(device)
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1).to(device)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update.view(input.size(0), self.num_relation, self.input_dim).permute(1, 0, 2).reshape(self.num_relation*input.size(0), self.input_dim)

    def combine(self, input, update):
        # 自环特征
        device = input.device
        self.linear.to(device)  # Ensure the linear layers are on the correct device
        self.self_loop.to(device)
        input = input.repeat(self.num_relation, 1).to(device)
        loop_update = self.self_loop(input).to(device)
        
        output = self.linear(update)+loop_update
        if self.batch_norm:
            self.batch_norm.to(device)
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def forward(self, graph, input):
        device = input.device
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph.to(device), input)
        output = self.combine(input, update)
        return output


@R.register("layer.relationalGraphStack")
class relationalGraphStack(nn.Module):
    
    def __init__(self, dims, num_relation, edge_input_dim=None, batch_norm=True, activation="relu"):
        super(relationalGraphStack, self).__init__()
        self.num_layers = len(dims) - 1
        self.layers = nn.ModuleList()
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                self.layers.append(relationalGraphConv(dims[i], dims[i + 1], num_relation, edge_input_dim, batch_norm, activation))
            
        self.layers.append(relationalGraph(dims[-2], dims[-1], num_relation, edge_input_dim, batch_norm, activation))
            

    def forward(self, graph, input, new_edge_list=None):
        device = input.device
        x = input
        for layer in self.layers:
            x = layer(graph.to(device), x)         
        return x.reshape(graph.num_relation, input.size(0), -1)



#===================================================================================================================================
@R.register("layer.Rewirescorelayer")
class Rewirescorelayer(nn.Module):
    def __init__(self, in_features, out_features, num_relations,num_heads, window_size, k, temperature=0.5):
        super(Rewirescorelayer, self).__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.window_size = window_size
        self.k = k
        self.temperature = temperature
        
        self.query = nn.Linear(in_features, out_features * num_heads)
        self.key = nn.Linear(in_features, out_features * num_heads)
        self.scale = 1 / (out_features ** 0.5)
    
    
    
    class LocalAttention(nn.Module):
        def __init__(
            self,
            window_size,
            look_backward = 1,
            look_forward = None,
            dropout = 0.,
            dim = None,
            scale = None,
            pad_start_position = None
        ):
            super().__init__()

            self.scale = scale

            self.window_size = window_size

            self.look_backward = look_backward
            self.look_forward = look_forward
            
            self.dropout = nn.Dropout(dropout)
            self.pad_start_position = pad_start_position
            
            

        def exists(self,val):
            return val is not None

        # 如果value不存在，返回d
        def default(self,value, d):
            return d if not self.exists(value) else value

        def to(self, t):
            return {'device': t.device, 'dtype': t.dtype}

        def max_neg_value(self, tensor):
            return -torch.finfo(tensor.dtype).max  #返回给定张量数据类型的所能表示的最大负值

        def look_around(self, x, backward = 1, forward = 0, pad_value = -1, dim = 2):  #x = bk: (40, 32, 16, 64)
            t = x.shape[1]    #获取一共有多少个窗口，这里是32
            dims = (len(x.shape) - dim) * (0, 0)   #一个长度为 len(x.shape) - dim 的元组，每个元素为 (0, 0)；其中len(x.shape) = 4
            padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)   #在第二维度上，前面加backward个元素，后面加forward个元素 -> (40, 33, 16, 64)
            tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)] #一个张量列表，每个张量的维度为(40, 32, 16, 64), len = 2
            return torch.cat(tensors, dim = dim) #在第二维度上拼接 -> (40, 32, 32, 64)
    
        def forward(
            self,
            q, k,
            mask = None,
            input_mask = None,
            window_size = None
        ):

            mask = self.default(mask, input_mask)
            assert not (self.exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'
            shape, pad_value, window_size, look_backward, look_forward = q.shape, -1, self.default(window_size, self.window_size), self.look_backward, self.look_forward
            (q, packed_shape), (k, _) = map(lambda t: pack([t], '* n d'), (q, k))  #打包成[5, 8, 512, 64] -> [40, 512, 64] 


            b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype   # 40, 512, 64
            scale = self.default(self.scale, dim_head ** -0.5)
            assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

            windows = n // window_size  # 512 / 16 = 32

            seq = torch.arange(n, device = device)                  # 0, 1, 2, 3, ..., 511
            b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)    # (1, 32, 16) 排序序列变形后的矩阵

            # bucketing

            bq, bk = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k)) #重构：（40，512，64）->（40, 32, 16, 64）

            bq = bq * scale    # (40, 32, 16, 64)
 
            look_around_kwargs = dict(
                backward =  look_backward,
                forward =  look_forward,
                pad_value = pad_value
            )

            bk = self.look_around(bk, **look_around_kwargs)      # (40, 32, 32, 64)
    

            # calculate positions for masking

            bq_t = b_t
            bq_k = self.look_around(b_t, **look_around_kwargs) # (1, 32, 32)

            bq_t = rearrange(bq_t, '... i -> ... i 1')      # (1, 32, 16, 1)
            bq_k = rearrange(bq_k, '... j -> ... 1 j')      # (1, 32, 1, 16)

            pad_mask = bq_k == pad_value

            sim = torch.einsum('b h i e, b h j e -> b h i j', bq, bk)  # (40, 32, 16, 64) * (40, 32, 32, 64) -> (40, 32, 16, 32)

            mask_value = self.max_neg_value(sim)

            sim = sim.masked_fill(pad_mask, mask_value)


            if self.exists(mask):
                batch = mask.shape[0]    # 5
                assert (b % batch) == 0

                h = b // mask.shape[0]  # 8

                mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
                mask = self.look_around(mask, **{**look_around_kwargs, 'pad_value': False})
                mask = rearrange(mask, '... j -> ... 1 j')
                mask = repeat(mask, 'b ... -> (b h) ...', h = h)

                sim = sim.masked_fill(~mask, mask_value)
                del mask
                
            indices = [self.pad_start_position[i] // window_size for i in range(len(self.pad_start_position)) if i % 2 != 0]
            all_indices = list(range(windows))
            remaining_indices = [idx for idx in all_indices if idx not in indices]
            
            # 使用剩余的索引选择元素
            rest_sim = sim[:, remaining_indices, :, :]

            # attention
            attn = rest_sim.softmax(dim = -1)
            attn = self.dropout(attn)
            
            return attn

    def insert_zero_rows(self, tensor, lengths, target_lengths):
        assert len(lengths) == len(target_lengths), "Lengths and target lengths must be of the same length."
        
        # 计算每个位置需要插入的零行数
        zero_rows = [target - length for length, target in zip(lengths, target_lengths)]
        
        # 初始化结果列表
        parts = []
        mask_parts = []
        start = 0
        
        for i, length in enumerate(lengths):
            end = start + length
            
            # 原始张量部分
            parts.append(tensor[:, start:end, :])
            mask_parts.append(torch.ones(tensor.size(0), length, dtype=torch.bool, device=tensor.device))
            
            # 插入零行
            if zero_rows[i] > 0:
                zero_padding = torch.zeros(tensor.size(0), zero_rows[i], tensor.size(2), device=tensor.device)
                mask_padding = torch.zeros(tensor.size(0), zero_rows[i], dtype=torch.bool, device=tensor.device)
                parts.append(zero_padding)
                mask_parts.append(mask_padding)
            
            start = end
        
        # 拼接所有部分
        padded_tensor = torch.cat(parts, dim=1)
        mask = torch.cat(mask_parts, dim=1)
        
        return padded_tensor, mask


    def round_up_to_nearest_k_and_a_window_size(self, lst, k):
        pad_start_position = []
        result_lst = [(x + k - 1) // k * k +k for x in lst]
        for i in range(len(lst)):
            pad_start_position.append(sum(result_lst[:i])-i*k + lst[i])
            pad_start_position.append(sum(result_lst[:i+1])-k)
        return result_lst, pad_start_position

        
    def displace_tensor_blocks_to_rectangle(self, tensor, displacement):
        batch_size, num_blocks, block_height, block_width = tensor.shape

        # 计算新矩阵的宽度和高度
        height = num_blocks * displacement
        width =  (2 + num_blocks) * displacement

        # 初始化新的大张量，确保其形状为 (batch_size, height, width)
        new_tensor = torch.zeros(batch_size, height, width, device=tensor.device, dtype=tensor.dtype)

        for i in range(num_blocks):
            start_pos_height = i * displacement
            start_pos_width = i * displacement
            end_pos_height = start_pos_height + block_height
            end_pos_width = start_pos_width + block_width

            new_tensor[:, start_pos_height:end_pos_height, start_pos_width:end_pos_width] = tensor[:, i, :, :]

        return new_tensor
    
    def forward(self, graph, node_features, sample=False):
        
        device = node_features.device
        num_relation = self.num_relations
        index = graph.num_nodes.tolist()
        
        target_input, pad_start_position = self.round_up_to_nearest_k_and_a_window_size(index, self.window_size)
        padding_input, mask = self.insert_zero_rows(node_features, index, target_input)
        
        self.query = self.query.to(device)
        self.key = self.key.to(device)
        Q = self.query(padding_input).view(num_relation, padding_input.size(1), self.num_heads, self.output_dim).permute(0, 2, 1, 3)                           # (num_relations, num_nodes, num_heads, out_features
        K = self.key(padding_input).view(num_relation, padding_input.size(1), self.num_heads, self.output_dim).permute(0, 2, 1, 3)                             # (num_relations, num_nodes, num_heads, out_features)
        Q = Q.reshape(num_relation * self.num_heads, padding_input.size(1), self.output_dim)                                                  # (num_relations*num_heads, num_nodes, out_features)
        K = K.reshape(num_relation * self.num_heads, padding_input.size(1), self.output_dim) 
        
        attn = self.LocalAttention(
            dim = self.output_dim,                   # dimension of each head (you need to pass this in for relative positional encoding)
            window_size = self.window_size,          # window size. 512 is optimal, but 256 or 128 yields good enough results
            look_backward = 1,                  # each window looks at the window before
            look_forward = 1,                   # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
            dropout = 0.1,
            pad_start_position = pad_start_position
            
        ) 
        
        attn = attn(Q, K, mask = mask).view(num_relation, self.num_heads, -1, self.window_size, 3*self.window_size).mean(dim=1) 
        
        if sample ==True:
            attn = attn.reshape(-1, self.window_size, 3*self.window_size)

            # 确保 -LARGE_NUMBER 的类型与 result_tensor 相同
            large_negative_number = torch.tensor(-LARGE_NUMBER, dtype=attn.dtype, device=attn.device)
            score = torch.where(attn==0, large_negative_number, attn)
            model = GumbleSampler(self.k, tau=1, hard=True)
            score = model(score)
            val = model.validate(score)
            
            accuracy = (score == val).float().mean().item()
            #print(f'Accuracy: {accuracy:.4f}')
            
            score = score.reshape(num_relation, -1, self.window_size, 3*self.window_size)
        else:
            score = attn
            
        result_tensor = self.displace_tensor_blocks_to_rectangle(score, self.window_size)
        result_tensor = result_tensor[:, :, self.window_size:-self.window_size]
        indice = [pad_start_position[i] for i in range(len(pad_start_position)) if i % 2 == 0]
        indices = []

        for num in indice:
            next_multiple_of_window_size = ((num + self.window_size-1) // self.window_size) * self.window_size  # 计算向上取10的倍数
            sequence = range(num, next_multiple_of_window_size)  # 生成序列
            indices.extend(sequence)  # 直接将序列中的元素添加到结果列表中
        all_indices = list(range(result_tensor.size(1)))
        remaining_indices = [idx for idx in all_indices if idx not in indices]
        
        result_tensor = result_tensor[:, remaining_indices, :]
        result_tensor = result_tensor[:, :, remaining_indices]

        
        return result_tensor.permute(1, 0, 2).contiguous().view(result_tensor.size(1), result_tensor.size(0)*result_tensor.size(2))
    
#===================================================================================================================================
class GraphRewiring(nn.Module, core.Configurable):

    max_seq_dist = 10

    def __init__(self,  edge_feature="gearnet"):
        super(GraphRewiring, self).__init__()
        
        self.edge_feature = edge_feature
        
    
    def adjacency_matrix_to_edge_list(self, A):
        # 找到非零元素的索引
        idx = torch.nonzero(A, as_tuple=False) 
        i = idx[:, 0]  
        j = idx[:, 1]
        line_num = A.size(0)  
        
        # 计算关系编号（relation）和调整后的列索引（j_prime）
        relation = torch.div(j, line_num, rounding_mode='floor')  # 关系编号，0到6
        if torch.any(relation > 6):
                    raise ValueError("张量中存在大于7的元素！")
        j_prime = j % line_num  # 调整后的列索引，0到184
        
        # 组合得到edge_list矩阵，形状为[n, 3]
        edge_list = torch.stack([i, j_prime, relation], dim=1)
        
        # 根据edge_list的索引顺序，获取对应的edge_weighted矩阵
        edge_weighted = A[i, j]
        
        return edge_list, edge_weighted

    

    def edge_gearnet(self, graph, edge_list, num_relation):
        node_in, node_out, r = edge_list.t()
        residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        in_residue_type = graph.residue_type[residue_in]
        out_residue_type = graph.residue_type[residue_out]
        sequential_dist = torch.abs(residue_in - residue_out)
        spatial_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)

        return torch.cat([
            functional.one_hot(in_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(out_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(r, num_relation),
            functional.one_hot(sequential_dist.clamp(max=self.max_seq_dist), self.max_seq_dist + 1),
            spatial_dist.unsqueeze(-1)
        ], dim=-1)

    def apply_node_layer(self, graph):
        
        return graph

    def apply_edge_layer(self, graph, edge_list, edge_weighted):
        
        node_in = edge_list[:, 0]
        edge2graph = graph.node2graph[node_in]
        order = edge2graph.argsort()
        edge_list = edge_list[order]
        edge_weighted = edge_weighted[order]
        num_edges = edge2graph.bincount(minlength=graph.batch_size)
        offsets = (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(num_edges)
        num_relation = graph.num_relation

        if hasattr(self, "edge_%s" % self.edge_feature):
            edge_feature = getattr(self, "edge_%s" % self.edge_feature)(graph, edge_list, num_relation)
        elif self.edge_feature is None:
            edge_feature = None
        else:
            raise ValueError("Unknown edge feature `%s`" % self.edge_feature)
        data_dict, meta_dict = graph.data_by_meta(include=(
            "node", "residue", "node reference", "residue reference", "graph"
        ))

        if isinstance(graph, data.PackedProtein):
            data_dict["num_residues"] = graph.num_residues
        if isinstance(graph, data.PackedMolecule):
            data_dict["bond_type"] = torch.zeros_like(edge_list[:, 2])
        return type(graph)(edge_list, num_nodes=graph.num_nodes, num_edges=num_edges, num_relation=num_relation,
                           view=graph.view, offsets=offsets, edge_feature=edge_feature, edge_weight=edge_weighted,
                           meta_dict=meta_dict, **data_dict)

    def forward(self, graph, attn_output):
        """
        Generate a new graph based on the input graph and pre-defined node and edge layers.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            graph (Graph): new graph(s)
        """
        device = graph.device
        graph = self.apply_node_layer(graph)
        edge_list, edge_weighted = self.adjacency_matrix_to_edge_list(attn_output)
        graph = self.apply_edge_layer(graph, edge_list.to(device), edge_weighted.to(device))
        return graph

  
#============================================================================================================================
# 可rewire的几何关系图卷积
@R.register("layer.RewireGearnet")
class RewireGeometricRelationalGraphConv(relationalGraphConv):
    gradient_checkpoint = False

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(RewireGeometricRelationalGraphConv, self).__init__(input_dim, output_dim, num_relation, edge_input_dim,
                                                                    batch_norm, activation)
        self.num_relation = num_relation
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.activation = getattr(F, activation) if activation else None
        self.edge_linear = nn.Linear(edge_input_dim, output_dim) if edge_input_dim else None

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation
        device = input.device  
        
        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), graph.edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t(), input.to(device))
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update
            
        return update.view(input.size(0), self.num_relation * self.input_dim).to(device)

    

@R.register("models.RewireGearnet")
class RewireGearnet(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(RewireGearnet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(RewireGeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(RewireGeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        
        device = input.device
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                     dim_size=graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                self.batch_norms[i].to(device)
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        } 