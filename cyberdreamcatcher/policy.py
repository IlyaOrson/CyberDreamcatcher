from collections import namedtuple

import torch
from torch.nn import ModuleDict
from torch.distributions import Categorical

from cyberdreamcatcher.utils import ravel_multi_index
from cyberdreamcatcher.gnn import GATGlobalConv


class ActionLogits:
    """Handles the conversion between the logits taken from outputs per node and the action space.
    Global action logits are the sum of the last 2 columns of the per-node action logits.
    """

    def __init__(self, action_logits, mask_node=None):
        self._raw_logits = action_logits

        self.node_logits = action_logits[:, :-2]

        if mask_node is not None:
            assert 0 <= mask_node < self.node_logits.shape[0], "Invalid mask_node"
            # Create a mask of the same shape as the tensor
            mask = torch.zeros_like(self.node_logits, dtype=torch.bool)
            mask[mask_node, :] = True  # Set True for the entire row you want to mask
            # Apply masked_fill
            self.node_logits = self.node_logits.masked_fill(mask, float("-inf"))

        # TODO mask sleep action and use sum for monitor?
        self.sleep_logit = torch.mean(action_logits[:, -1]).unsqueeze(-1)
        self.monitor_logit = torch.mean(action_logits[:, -2]).unsqueeze(-1)

        self.flat_logits = torch.cat(
            (self.sleep_logit, self.monitor_logit, self.node_logits.flatten())
        )

    def flat_to_multidim(self, action_flat):
        # The first entry of action represents the host
        # so it is irrelevant for global actions
        # [x, 0] == "Sleep"
        # [x, 1] == "Monitor"
        if action_flat == 0:  # Sleep
            action = [0, 0]
        elif action_flat == 1:  # Monitor
            action = [0, 1]
        else:
            # Recover the corresponding multidimensional index from the flattened action
            # Remove the global actions from the flattened index
            host_id, action_id = torch.unravel_index(
                action_flat - 2, self.node_logits.shape
            )
            action = (host_id, action_id + 2)  # Add 2 to account for global actions

        return torch.tensor(action, device=action_flat.device)

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
            # Calculate flat index within node_logits and add 2 for the final flat index
            action_flat = ravel_multi_index(shifted_action, self.node_logits.shape) + 2

        return action_flat.to(action_multi.device)


class Police(torch.nn.Module):
    "Defensive blue agent - TacticsAI GAT"

    PoliceReport = namedtuple(
        "PoliceReport", ["action", "log_prob", "entropy", "value", "attention"]
    )

    def __init__(
        self,
        env,
        latent_node_dim=None,
        train_critic=False,
        actor_heads=1,
        critic_heads=1,
        mask_node="User0",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mask_node = env.host_enumeration[mask_node]

        if latent_node_dim is None:
            latent_node_dim = env.host_encoding_dim

        # Latent layers (typically 1-4 in gnns due to oversmoothing)
        self.actor_latent_0 = GATGlobalConv(
            in_channels=env.host_encoding_dim,
            out_channels=latent_node_dim,
            global_dim=env.global_encoding_dim,
            edge_dim=env.edge_encoding_dim,
            heads=actor_heads,
            share_weights=False,
        )
        self.actor_latent_1 = GATGlobalConv(
            in_channels=actor_heads * latent_node_dim,
            out_channels=latent_node_dim,
            global_dim=env.global_encoding_dim,
            edge_dim=env.edge_encoding_dim,
            heads=actor_heads,
            share_weights=False,
        )
        # Returns logits in a matrix of shape (nodes x actions)
        self.actor_head = GATGlobalConv(
            in_channels=actor_heads * latent_node_dim,
            out_channels=env.num_actions,  # one score per host/node and per action
            global_dim=env.global_encoding_dim,
            edge_dim=env.edge_encoding_dim,
            heads=1,
            concat=False,  # average instead of concat
            share_weights=False,
        )
        self.actor_layers = ModuleDict(
            {
                "latent_0": self.actor_latent_0,
                "latent_1": self.actor_latent_1,
                "head": self.actor_head,
            }
        )

        # NOTE: this may break backwards compatibility
        # since previous trained policies have an unused critic
        self.train_critic = train_critic

        # Train critic only in actor-critic methods
        if self.train_critic:
            self.critic_latent_0 = GATGlobalConv(
                in_channels=env.host_encoding_dim,
                out_channels=latent_node_dim,
                global_dim=env.global_encoding_dim,
                edge_dim=env.edge_encoding_dim,
                heads=critic_heads,
                share_weights=False,
            )
            self.critic_latent_1 = GATGlobalConv(
                in_channels=latent_node_dim,
                out_channels=latent_node_dim,
                global_dim=env.global_encoding_dim,
                edge_dim=env.edge_encoding_dim,
                heads=critic_heads,
                share_weights=False,
            )
            self.critic_head = GATGlobalConv(
                in_channels=latent_node_dim,
                out_channels=1,  # one score per node
                global_dim=env.global_encoding_dim,
                edge_dim=env.edge_encoding_dim,
                heads=critic_heads,
                share_weights=False,
            )

            self.critic_layers = ModuleDict(
                {
                    "latent_0": self.critic_latent_0,
                    "latent_1": self.critic_latent_1,
                    "head": self.critic_head,
                }
            )

        # Train critic only in actor-critic methods
        # if not self.train_critic:
        #     for param in self.critic_layers.parameters():
        #         param.requires_grad = False

    def actor(self, nodes_matrix, edge_index, edge_matrix, global_matrix, return_attention_weights=False):
        # Score each node to select actions
        actor_latent_nodes = self.actor_latent_0(
            nodes_matrix,
            edge_index,
            edge_attr=edge_matrix,
            global_attr=global_matrix,
        )
        actor_latent_nodes = self.actor_latent_1(
            actor_latent_nodes,
            edge_index,
            edge_attr=edge_matrix,
            global_attr=global_matrix,
        )
        # In case we want to visualise the attention weights
        if return_attention_weights:
            action_logits, attention_weights = self.actor_head(
                actor_latent_nodes,
                edge_index,
                edge_attr=edge_matrix,
                global_attr=global_matrix,
                return_attention_weights=True,
            )
            return action_logits, attention_weights

        action_logits = self.actor_head(
            actor_latent_nodes,
            edge_index,
            edge_attr=edge_matrix,
            global_attr=global_matrix,
        )
        return action_logits, None

    def critic(self, nodes_matrix, edge_index, edges_matrix, global_matrix):
        # Score each node to value state
        critic_latent_nodes = self.critic_latent_0(
            nodes_matrix,
            edge_index,
            edge_attr=edges_matrix,
            global_attr=global_matrix,
        )
        critic_latent_nodes = self.critic_latent_1(
            critic_latent_nodes,
            edge_index,
            edge_attr=edges_matrix,
            global_attr=global_matrix,
        )
        node_values = self.critic_head(
            critic_latent_nodes,
            edge_index,
            edge_attr=edges_matrix,
            global_attr=global_matrix,
        )
        value = torch.sum(node_values)
        return value

    def get_action_logits(self, graph, return_attention_weights=False):
        # Destructure Data() object from pytorch geometric
        nodes_matrix = graph.x
        edge_index = graph.edge_index
        edges_matrix = graph.edge_attr

        global_matrix = graph.get("global_attr", None)

        action_logits, attention = self.actor(
            nodes_matrix,
            edge_index,
            edges_matrix,
            global_matrix,
            return_attention_weights=return_attention_weights,
        )

        if self.train_critic:
            value = self.critic(
                nodes_matrix,
                edge_index,
                edges_matrix,
                global_matrix,
            )
        else:
            value = None

        return ActionLogits(action_logits, mask_node=self.mask_node), value, attention

    def forward(self, graph, action=None, return_attention_weights=False):
        action_logits, value, attention = self.get_action_logits(
            graph, return_attention_weights=return_attention_weights
        )

        # Flatten the logits array to use a one-dimensional categorical distribution.
        # distribution = Categorical(logits=action_logits.flatten())
        distribution = Categorical(logits=action_logits.flat_logits)
        entropy = distribution.entropy()

        if action is None:
            # Sample action and return it as multidimensional version

            # distribution.mode()  # deterministic
            action_flat = distribution.sample()  # stochastic

            action = action_logits.flat_to_multidim(action_flat)
            assert action_flat == action_logits.multidim_to_flat(action)
        else:
            # Convert multidimensional action to the corresponding flat action
            action_flat = action_logits.multidim_to_flat(action)
            assert torch.equal(action, action_logits.flat_to_multidim(action_flat))

        action_log_prob = distribution.log_prob(action_flat)

        return self.PoliceReport(action, action_log_prob, entropy, value, attention)
