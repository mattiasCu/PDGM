import math

import torch
from torch import nn

from torchdrug import core, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R

class GraphRewiring(nn.Module, core.Configurable):

    max_seq_dist = 10

    def __init__(self,  edge_feature="gearnet"):
        super(GraphRewiring, self).__init__()
        
        self.edge_feature = edge_feature

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

        #num_edges = edge_list.size(0)
        #num_edges = torch.tensor(num_edges, device=graph.device)
        #num_relations = torch.tensor(num_relations, device=graph.device)
        #num_relation = num_relations.sum()
        #offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
        #edge_list[:, 2] += offsets

        # reorder edges into a valid PackedGraph
        node_in = edge_list[:, 0]
        edge2graph = graph.node2graph[node_in]
        order = edge2graph.argsort()
        edge_list = edge_list[order]
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

    def forward(self, graph, edge_list=None, edge_weighted = None):
        """
        Generate a new graph based on the input graph and pre-defined node and edge layers.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            graph (Graph): new graph(s)
        """
        graph = self.apply_node_layer(graph)
        graph = self.apply_edge_layer(graph, edge_list, edge_weighted)
        return graph