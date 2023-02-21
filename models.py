from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseGraphConv, GraphConv
from torch_geometric.utils import to_dense_adj, to_dense_batch

from hoscpool import dense_hoscpool


class GNN(torch.nn.Module):
    """Graph Neural Network with graph pooling"""

    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        num_classes: int,
        hidden_dim: list = [32, 32],
        mlp_hidden_dim: int = 32,
        mu: float = 0.1,
        new_ortho: bool = False,
        cluster_ratio: float = 0.2,
        dropout: float = 0,
        device=None,
        sn=False,
    ):

        super(GNN, self).__init__()

        self.num_layers = len(hidden_dim)
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.rms = []
        self.mu = mu
        self.sn = sn
        self.new_ortho = new_ortho
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pooling_type = dense_hoscpool

        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(GraphConv(num_node_features, hidden_dim[i]))  # sparse
            else:
                self.convs.append(
                    DenseGraphConv(hidden_dim[i - 1], hidden_dim[i])
                )  # dense

        # Pooling layers (to learn cluster matrix)
        for i in range(self.num_layers - 1):
            num_nodes = ceil(cluster_ratio * num_nodes)  # K
            self.pools.append(Linear(hidden_dim[i], num_nodes))

        # Dense layers for prediction
        self.lin1 = Linear(hidden_dim[-1], mlp_hidden_dim)
        self.lin2 = Linear(mlp_hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, mask=None):

        # Normalise adj
        edge_weight = None

        ### Block 1
        # convolution
        x = F.relu(self.convs[0](x, edge_index))
        # convert batch sparse to batch dense
        x, mask = to_dense_batch(x, batch)
        # convert sparse adj to dense adj
        adj = to_dense_adj(edge_index, batch, edge_weight)
        # Cluster ass matrix
        s = self.pools[0](x)
        # Pooling
        x, adj, mc, o = self.pooling_type(
            x,
            adj,
            s,
            self.mu,
            alpha=0.5,
            new_ortho=self.new_ortho,
            mask=mask,
        )  # pooling

        ### Middle blocks
        for i in range(1, self.num_layers - 1):
            x = F.relu(self.convs[i](x, adj))  # Convolution
            s2 = self.pools[i](x)  # cluster ass matrix
            x, adj, mc_aux, o_aux = self.pooling_type(
                x,
                adj,
                s2,
                self.mu,
                alpha=0.5,
                new_ortho=self.new_ortho,
            )  # pooling

            mc += mc_aux
            o += o_aux

        ### Last block
        x = self.convs[self.num_layers - 1](x, adj)  # conv
        x = x.mean(dim=1)  # global pooling

        # Dense classifier
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        if self.num_layers > 2:
            # Complex archi: MP - Pooling - MP - Pooling - MP - Global Pooling - Dense (x2)
            return x, mc, o, [s, s2]
        else:
            # Simple archi: MP - Pooling - MP - Global Pooling - Dense (x2)
            return x, mc, o, s
