import torch
from torch import nn

from torch_geometric.nn.conv import GATv2Conv
# FIXME add global vector in attention mechanism
# from blueskynet.gnn import GATv2Conv

from stable_baselines3.common.distributions import CategoricalDistribution


class Police(nn.Module):
    def __init__(self, env, latent_node_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gnn_layer = GATv2Conv(
            in_channels=env.host_embedding_size,
            out_channels=latent_node_dim,
            # global_channels=env.global_embedding_size,
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=True,
        )
        # NOTE returns logits in a matrix of shape nodes x actions
        self.action_layer = GATv2Conv(
            in_channels=latent_node_dim,
            out_channels=env.num_actions,  # one score per host/node and per action
            # global_channels=env.global_embedding_size,
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=True,
        )

        self.action_dist = CategoricalDistribution(action_dim=1)

    def get_action_logits(self, graph):
        # Destructure Data() object from pytorch geometric
        nodes_matrix = graph.x
        edge_index = graph.edge_index
        edges_matrix = graph.edge_attr
        # global_vector = graph.global_attr

        # A few gnn layers to pass messages around the graph
        latent_nodes = self.gnn_layer(
            nodes_matrix, edge_index, edges_matrix
        )
        # latent_nodes = self.gnn_layer(
        #     nodes_matrix, edge_index, global_vector, edges_matrix
        # )

        # Use gnn to score each node to select an action
        action_logits = self.action_layer(
            latent_nodes, edge_index, edges_matrix
        )
        # action_logits = self.action_layer(
        #     latent_nodes, edge_index, global_vector, edges_matrix
        # )

        return action_logits

    def forward(self, graph):
        action_logits = self.get_action_logits(graph)

        # Turn logits into the categorical probability distribution
        # Flatten the logits array to use a one-dimensional distribution.
        action_selection_flat, action_log_prob = self.action_dist.log_prob_from_params(
            action_logits.flatten()
        )

        # Recover the original array index from the flattened selection
        action_selection = torch.unravel_index(
            action_selection_flat, action_logits.shape
        )

        # action_logits == action_logits.flatten().reshape(action_logits.shape)
        # assert (
        #     action_logits[action_selection]
        #     == action_logits.flatten()[action_selection_flat]
        # )

        return action_selection, action_log_prob


class ConditionalPolice(nn.Module):
    def __init__(self, env, latent_node_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO explore performance changes between out channels or heads as both define output dimension
        self.gnn_layer = GATv2Conv(
            in_channels=env.host_embedding_size,
            out_channels=latent_node_dim,
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=True,
        )

        # TODO add a couple of extra layers

        self.node_layer = GATv2Conv(
            in_channels=latent_node_dim,
            out_channels=1,  # one score per host node (choose a node to act on)
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=True,
        )
        self.action_layer = GATv2Conv(
            in_channels=latent_node_dim,
            out_channels=env.num_actions,
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=True,
        )

        self.node_dist = CategoricalDistribution(action_dim=1)
        self.action_dist = CategoricalDistribution(action_dim=1)

    def forward(self, graph):
        # Destructure Data() object from pytorch geometric
        nodes_matrix = graph.x
        edge_index = graph.edge_index
        edges_matrix = graph.edge_attr

        # A few gnn layers to pass messages around the graph
        latent_nodes = self.gnn_layer(nodes_matrix, edge_index, edges_matrix)

        # Use gnn to score each node to select an action
        node_logits = self.node_layer(latent_nodes, edge_index, edges_matrix)
        action_logits = self.action_layer(latent_nodes, edge_index, edges_matrix)

        # Turn logits into the categorical probability distribution
        node_selection, node_log_prob = self.node_dist.log_prob_from_params(
            node_logits.squeeze()
        )
        action_selection, action_log_prob = self.action_dist.log_prob_from_params(
            action_logits[node_selection, :]
        )

        return (node_selection, action_selection), node_log_prob + action_log_prob
