import torch
from torch.distributions import Categorical

# from torch_geometric.nn.conv import GATv2Conv
from blueskynet.gnn import GATGlobalConv


class Police(torch.nn.Module):
    def __init__(self, env, latent_node_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gnn_layer_0 = GATGlobalConv(
            in_channels=env.host_embedding_size,
            out_channels=latent_node_dim,
            global_channels=env.global_embedding_size,
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=False,
        )
        self.gnn_layer_1 = GATGlobalConv(
            in_channels=latent_node_dim,
            out_channels=latent_node_dim,
            global_channels=env.global_embedding_size,
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=False,
        )
        # NOTE returns logits in a matrix of shape (nodes x actions)
        self.action_layer = GATGlobalConv(
            in_channels=latent_node_dim,
            out_channels=env.num_actions,  # one score per host/node and per action
            global_channels=env.global_embedding_size,
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=True,
        )

        # self.action_dist = CategoricalDistribution(action_dim=1)

    def get_action_logits(self, graph):
        # Destructure Data() object from pytorch geometric
        nodes_matrix = graph.x
        edge_index = graph.edge_index
        edges_matrix = graph.edge_attr
        global_vector = graph.global_attr

        # A few gnn layers to pass messages around the graph
        # latent_nodes = self.gnn_layer(nodes_matrix, edge_index, edges_matrix)
        latent_nodes = self.gnn_layer_0(
            nodes_matrix, edge_index, global_vector, edges_matrix
        )
        latent_nodes = self.gnn_layer_1(
            latent_nodes, edge_index, global_vector, edges_matrix
        )

        # Use gnn to score each node to select an action
        # action_logits = self.action_layer(latent_nodes, edge_index, edges_matrix)
        action_logits = self.action_layer(
            latent_nodes, edge_index, global_vector, edges_matrix
        )

        return action_logits

    def forward(self, graph):
        action_logits = self.get_action_logits(graph)

        # Flatten the logits array to use a one-dimensional categorical distribution.
        distribution = Categorical(logits=action_logits.flatten())
        # distribution.mode()  # deterministic
        action_flat = distribution.sample()  # stochastic
        action_log_prob = distribution.log_prob(action_flat)

        # Recover the corresponding multidimensional index from the flattened one
        action = torch.unravel_index(action_flat, action_logits.shape)

        # action_logits == action_logits.flatten().reshape(action_logits.shape)
        # assert (
        #     action_logits[action]
        #     == action_logits.flatten()[action_flat]
        # )

        return action, action_log_prob
