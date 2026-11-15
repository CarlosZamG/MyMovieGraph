"""
This module defines the graph neural network architecture.

It consists of a homogeneous GraphSAGE encoder, an edge decoder for
link prediction, and a parent Model container that converts the encoder
to handle heterogeneous graph structures
"""

import torch

from torch_geometric.nn import SAGEConv, to_hetero

class GraphSageEncoder(torch.nn.Module):
    """
    Uses the initial input node features (both users and movies)
    and performs two GraphSAGE convolutions
    """
    def __init__(self, hidden_channels, out_channels, dropout):
        super().__init__()
        # Using (-1, -1) automatically infers the input feature dimensions
        # of target and source nodes at runtime during the first forward pass
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, project = True)
        self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, x, edge_index):
        """
        Performs the Message Passing.
        
        Each user node aggregates feature representations from the movies
        they positively rated (3.5 or higher), while each movie node
        aggregates features from all users who positively rated it. 
        
        Also aggregates 2-hop structural context. 
        """
        x = self.conv1(x, edge_index)
        x = self.norm1(x).relu() 
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    """
    Predicts the interaction affinity score between pairs of nodes
    given their latent node representations
    """
    def __init__(self):
        super().__init__()

    def forward(self, z_dict, edge_label_index):
        """
        Predict scores for target user-movie edges

        Args:
            - z_dict: A dictionary mapping node types ("user", "movie") to
            their respective node embedding tensors
            - edge_label_index: A 2D index tensor of shape [2, num_edges] 
            where row 0 contains target user node indices and row 1 contains 
            target movie node indices
        
        Returns:
            - The dot-product affinity scores for each requested edge pair
        """

        row, col = edge_label_index
        
        # Extract the embeddings for the specific users and movies in the edge index
        user_embeds = z_dict["user"][row]
        movie_embeds = z_dict["movie"][col]
        
        # Compute the dot product along the feature dimension (dim=-1)
        scores = (user_embeds * movie_embeds).sum(dim=-1)

        return scores

class Model(torch.nn.Module):
    """
    Parent container that converts the encoder to handle heterogeneous
    graph structures
    """
    def __init__(self, hidden_channels, data, dropout):
        super().__init__()
        self.encoder = GraphSageEncoder(hidden_channels, hidden_channels, dropout)
        # to_hetero() takes an existing GNN model and duplicates the 
        # message functions to work on each edge type individually
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')
        self.decoder = EdgeDecoder()
    
    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z_dict, edge_label_index):
        return self.decoder(z_dict, edge_label_index)
