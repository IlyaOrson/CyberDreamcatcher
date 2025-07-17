from collections import namedtuple

import torch
from torch.nn import ModuleDict
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv

from cyberdreamcatcher.utils import ravel_multi_index


class ActionLogits:
    """Handles the conversion between the logits taken from outputs per node and the action space.
    Global action logits are the sum of the last 2 columns of the per-node action logits.
    """

    def __init__(
        self,
        action_logits,
        mask_node=None,
        host_to_decoys=None,
        action_enumeration=None,
        host_enumeration=None,
    ):
        self._raw_logits = action_logits

        self.node_logits = action_logits[:, :-2]

        if mask_node is not None:
            assert 0 <= mask_node < self.node_logits.shape[0], "Invalid mask_node"
            # Create a mask of the same shape as the tensor
            mask = torch.zeros_like(self.node_logits, dtype=torch.bool)
            mask[mask_node, :] = True  # Set True for the entire row you want to mask
            # Apply masked_fill
            self.node_logits = self.node_logits.masked_fill(mask, float("-inf"))

        if host_to_decoys and action_enumeration:
            # Create a mask for the decoys
            action_mask = torch.zeros_like(self.node_logits, dtype=torch.bool)
            for host_name, available_decoys in host_to_decoys.items():
                host_idx = host_enumeration[host_name]
                for action_name, action_idx in action_enumeration.items():
                    if (
                        action_name.startswith("Decoy")
                        and action_name not in available_decoys
                    ):
                        # We substract 2 from the action_idx to account for the global actions
                        # which correspond to the first 2 indexes in the action enumeration
                        # but the final columns in the action logits
                        assert action_idx >= 2
                        action_mask[host_idx, action_idx - 2] = True

            # Apply the mask to the node_logits
            self.node_logits.masked_fill_(action_mask, float("-inf"))

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
        num_layers=2,
        share_weights=False,
        residual=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.host_to_decoys = env.host_to_decoys
        self.host_enumeration = env.host_enumeration
        self.action_enumeration = env.action_enumeration
        self.mask_node = env.host_enumeration[mask_node]

        if latent_node_dim is None:
            latent_node_dim = env.host_encoding_dim

        self.num_layers = num_layers
        self.latent_node_dim = latent_node_dim
        self.actor_heads = actor_heads
        self.share_weights = share_weights
        self.residual = residual

        # Create actor layers dynamically
        self.actor_layers = ModuleDict()
        
        # Create latent layers (typically 1-4 in gnns due to oversmoothing)
        for i in range(num_layers):
            if i == 0:
                # First layer: input is host encoding
                in_channels = env.host_encoding_dim
            else:
                # Subsequent layers: input is from previous layer with heads
                in_channels = actor_heads * latent_node_dim
                
            self.actor_layers[f"latent_{i}"] = GATv2Conv(
                in_channels=in_channels,
                out_channels=latent_node_dim,
                edge_dim=env.edge_encoding_dim,
                heads=actor_heads,
                share_weights=self.share_weights,
                residual=self.residual,
            )
        
        # Returns logits in a matrix of shape (nodes x actions)
        self.actor_layers["head"] = GATv2Conv(
            in_channels=actor_heads * latent_node_dim,
            out_channels=env.num_actions,  # one score per host/node and per action
            edge_dim=env.edge_encoding_dim,
            heads=1,
            concat=False,  # average instead of concat
            share_weights=self.share_weights,
            residual=self.residual,
        )

        # NOTE: this may break backwards compatibility
        # since previous trained policies have an unused critic
        self.train_critic = train_critic

        # Train critic only in actor-critic methods
        if self.train_critic:
            self.critic_heads = critic_heads
            self.critic_layers = ModuleDict()
            
            # Create critic layers dynamically
            for i in range(num_layers):
                if i == 0:
                    # First layer: input is host encoding
                    in_channels = env.host_encoding_dim
                else:
                    # Subsequent layers: input is from previous layer with heads
                    in_channels = critic_heads * latent_node_dim
                    
                self.critic_layers[f"latent_{i}"] = GATv2Conv(
                    in_channels=in_channels,
                    out_channels=latent_node_dim,
                    edge_dim=env.edge_encoding_dim,
                    heads=critic_heads,
                    share_weights=self.share_weights,
                    residual=self.residual,
                )
            
            # Critic head layer
            self.critic_layers["head"] = GATv2Conv(
                in_channels=critic_heads * latent_node_dim,
                out_channels=1,  # one score per node
                edge_dim=env.edge_encoding_dim,
                heads=critic_heads,
                share_weights=self.share_weights,
                residual=self.residual,
            )

        # Train critic only in actor-critic methods
        # if not self.train_critic:
        #     for param in self.critic_layers.parameters():
        #         param.requires_grad = False

    def actor(
        self,
        nodes_matrix,
        edge_index,
        edge_matrix,
        return_attention_weights=False,
    ):
        # Score each node to select actions
        actor_latent_nodes = nodes_matrix
        
        # Forward pass through all latent layers
        for i in range(self.num_layers):
            actor_latent_nodes = self.actor_layers[f"latent_{i}"](
                actor_latent_nodes,
                edge_index,
                edge_attr=edge_matrix,
            )
        
        # In case we want to visualise the attention weights
        if return_attention_weights:
            action_logits, attention_weights = self.actor_layers["head"](
                actor_latent_nodes,
                edge_index,
                edge_attr=edge_matrix,
                return_attention_weights=True,
            )
            return action_logits, attention_weights

        action_logits = self.actor_layers["head"](
            actor_latent_nodes,
            edge_index,
            edge_attr=edge_matrix,
        )
        return action_logits, None

    def critic(
        self,
        nodes_matrix,
        edge_index,
        edges_matrix,
    ):
        # Score each node to value state
        critic_latent_nodes = nodes_matrix
        
        # Forward pass through all latent layers
        for i in range(self.num_layers):
            critic_latent_nodes = self.critic_layers[f"latent_{i}"](
                critic_latent_nodes,
                edge_index,
                edge_attr=edges_matrix,
            )
        
        node_values = self.critic_layers["head"](
            critic_latent_nodes,
            edge_index,
            edge_attr=edges_matrix,
        )
        value = torch.sum(node_values)
        return value

    def get_action_logits(self, graph, return_attention_weights=False):
        # Destructure Data() object from pytorch geometric
        nodes_matrix = graph.x
        edge_index = graph.edge_index
        edges_matrix = graph.edge_attr

        action_logits, attention = self.actor(
            nodes_matrix,
            edge_index,
            edges_matrix,
            return_attention_weights=return_attention_weights,
        )

        if self.train_critic:
            value = self.critic(
                nodes_matrix,
                edge_index,
                edges_matrix,
            )
        else:
            value = None

        return (
            ActionLogits(
                action_logits,
                mask_node=self.mask_node,
                host_to_decoys=self.host_to_decoys,
                action_enumeration=self.action_enumeration,
                host_enumeration=self.host_enumeration,
            ),
            value,
            attention,
        )

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
