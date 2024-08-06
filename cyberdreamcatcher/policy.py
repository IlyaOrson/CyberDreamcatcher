from collections import namedtuple

import torch
from torch.distributions import Categorical

from cyberdreamcatcher.utils import ravel_multi_index
from cyberdreamcatcher.gnn import GATGlobalConv


class ActionLogits:
    """Handles the conversion between the logits taken from outputs per node and the action space.
    Global action logits are the sum of the last 2 columns of the per-node action logits.
    """

    def __init__(self, action_logits):
        self._raw_logits = action_logits

        self.node_logits = action_logits[:, :-2]

        self.sleep_logit = torch.mean(action_logits[:, -1]).unsqueeze(-1)
        self.monitor_logit = torch.mean(action_logits[:, -2]).unsqueeze(-1)

        self.flat_logits = torch.cat(
            (self.sleep_logit, self.monitor_logit, self.node_logits.flatten())
        )

    def flat_to_multidim(self, action_flat):
        # The first entry of action represents the host
        # so it is irrelevant for these global actions
        if action_flat == 0:  # Sleep
            action = [0, 0]
        elif action_flat == 1:  # Monitor
            action = [0, 1]
        else:
            # Recover the corresponding multidimensional index from the flattened action
            # action = torch.unravel_index(action_flat, action_logits.shape)
            action = torch.unravel_index(action_flat - 2, self.node_logits.shape)

        return torch.tensor(action)

    def multidim_to_flat(self, action_multi):
        # Convert multidimensional action to the corresponding flat action
        assert len(action_multi) == self.node_logits.dim()
        if action_multi[-1] == 0:  # Sleep
            action_flat = torch.tensor(0)
        elif action_multi[-1] == 1:  # Monitor
            action_flat = torch.tensor(1)
        else:
            # This mutation is problematic when an action is provided and
            # only its log_prob is of interest to be calculated
            # action_multi[-1] -= 2
            shifted_action = torch.tensor([action_multi[0], action_multi[-1] - 2])
            action_flat = ravel_multi_index(shifted_action, self.node_logits.shape)

        return action_flat


class Police(torch.nn.Module):
    "Defensive blue agent - TacticsAI GAT"

    PoliceReport = namedtuple(
        "PoliceReport", ["action", "log_prob", "entropy", "value"]
    )

    def __init__(self, env, latent_node_dim, train_critic=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Latent layers (typically 1-4 in gnns due to oversmoothing)
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

        # Returns logits in a matrix of shape (nodes x actions)
        self.action_layer = GATGlobalConv(
            in_channels=latent_node_dim,
            out_channels=env.num_actions,  # one score per host/node and per action
            global_channels=env.global_embedding_size,
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=True,
        )
        self.critic_layer = GATGlobalConv(
            in_channels=latent_node_dim,
            out_channels=1,  # one score per node
            global_channels=env.global_embedding_size,
            edge_dim=env.edge_embedding_size,
            heads=1,
            share_weights=False,
        )
        # To train the critic only makes sense in actor-critic methods
        if not train_critic:
            for param in self.critic_layer.parameters():
                param.requires_grad = False

    def get_action_logits(self, graph):
        # Destructure Data() object from pytorch geometric
        nodes_matrix = graph.x
        edge_index = graph.edge_index
        edges_matrix = graph.edge_attr
        global_vector = graph.global_attr

        # A few gnn layers to pass messages around the graph
        latent_nodes = self.gnn_layer_0(
            nodes_matrix, edge_index, global_vector, edges_matrix
        )
        latent_nodes = self.gnn_layer_1(
            latent_nodes, edge_index, global_vector, edges_matrix
        )

        # Use gnn to score each node to select actions
        action_logits = self.action_layer(
            latent_nodes, edge_index, global_vector, edges_matrix
        )

        # with torch.no_grad():
        node_values = self.critic_layer(
            latent_nodes, edge_index, global_vector, edges_matrix
        )
        value = torch.sum(node_values)

        return ActionLogits(action_logits), value

    def forward(self, graph, action=None):
        action_logits, value = self.get_action_logits(graph)

        # Flatten the logits array to use a one-dimensional categorical distribution.
        # distribution = Categorical(logits=action_logits.flatten())
        distribution = Categorical(logits=action_logits.flat_logits)
        entropy = distribution.entropy()

        if action is None:
            # Sample action and return it as multidimensional version

            # distribution.mode()  # deterministic
            action_flat = distribution.sample()  # stochastic

            action = action_logits.flat_to_multidim(action_flat)
        else:
            # Convert multidimensional action to the corresponding flat action
            action_flat = action_logits.multidim_to_flat(action)

        action_log_prob = distribution.log_prob(action_flat)

        return self.PoliceReport(action, action_log_prob, entropy, value)
