from pathlib import Path

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import networkx as nx
import numpy as np

import torch
from torch.nn.functional import softmax


def plot_feasible_connections(
    env, axis=None, multipartite=False, show=False, block=False, filepath=None
):
    # Simplest version
    # graph = nx.DiGraph()
    # graph.add_nodes_from(env.host_names)
    # graph.add_edges_from(env.feasible_connections)
    # positions = nx.kamada_kawai_layout(graph)
    # nx.draw_networkx(graph, pos=positions)
    # plt.show()

    graph = nx.DiGraph()

    n_colors = len(env.subnet_names)
    cmap = plt.cm.Set2
    # cmap = plt.cm.get_cmap("gnuplot")
    colors = cmap(np.linspace(0, 1, n_colors))
    subnet_color_map = {
        subnet: colors[idx] for idx, subnet in enumerate(env.subnet_names)
    }

    for subnet, hostnames in env.subnet_hostnames_map.items():
        graph.add_nodes_from(hostnames, subnet=subnet)
    graph.add_edges_from(env.feasible_connections)

    if multipartite:
        node_positions = nx.multipartite_layout(
            graph, subset_key="subnet", align="horizontal"
        )
    else:
        node_positions = nx.kamada_kawai_layout(graph)

    if axis is None:
        _, axis = plt.subplots()
    else:
        axis.cla()

    axis.axis("off")

    for subnet, hostnames in env.subnet_hostnames_map.items():
        color = subnet_color_map[subnet]
        node_colors = len(hostnames) * [color]

        nx.draw_networkx_nodes(
            graph,
            pos=node_positions,
            ax=axis,
            nodelist=hostnames,
            node_color=node_colors,
            alpha=0.5,
            node_shape="D",
            node_size=400,
            label=subnet,
            linewidths=2,
            # edgecolors="grey",
            # cmap=cmap
        )

    nx.draw_networkx_edges(
        graph,
        pos=node_positions,
        ax=axis,
        alpha=0.25,
        connectionstyle="Arc3, rad = 0.3",
    )
    nx.draw_networkx_labels(graph, pos=node_positions, ax=axis, font_size=8, alpha=0.5)

    plt.title(f"{env.scenario_name}")
    plt.legend(markerscale=0.5)
    plt.tight_layout()
    if show:
        plt.show(block=block)

    if filepath:
        plt.savefig(Path(filepath), dpi=600)

    return node_positions


def _plot_observation(
    graph, node_positions=None, axis=None, show=False, block=False, cmap="plasma"
):
    if node_positions is None:
        node_positions = nx.kamada_kawai_layout(graph)

    if axis is None:
        _, axis = plt.subplots()
    else:
        axis.cla()

    cmap = get_cmap(cmap)
    node_color_prop = "num_local_ports"
    edge_color_prop = "connections"
    max_ports = 15
    max_connections = 15

    clean_nodes = []
    clean_colors = []
    malware_nodes = []
    malware_colors = []
    for node, props in graph.nodes().items():
        # If props is null it means this node was added by networkx
        # as an artifact to hold a connection from an added node to
        # a non existant node, so we can safely ignore it.
        if not props:
            continue

        if props["malware"] == 0:
            clean_nodes.append(node)
            clean_colors.append(int(props[node_color_prop]))

        elif props["malware"] == 1:
            malware_nodes.append(node)
            malware_colors.append(int(props[node_color_prop]))
        else:
            raise ValueError(f"Unexpected malware value {node['malware']}")

    nx.draw_networkx_nodes(
        graph,
        pos=node_positions,
        ax=axis,
        nodelist=clean_nodes,
        node_shape="o",
        node_color=clean_colors,
        cmap=cmap,
        vmin=0,
        vmax=max_ports,
        alpha=0.3,
    )
    nx.draw_networkx_nodes(
        graph,
        pos=node_positions,
        ax=axis,
        nodelist=malware_nodes,
        node_shape="X",
        node_color=malware_colors,
        cmap=cmap,
        vmin=0,
        vmax=max_ports,
        alpha=0.3,
    )

    edges = []
    edge_colors = []
    for edge, props in graph.edges().items():
        edges.append(edge)
        edge_colors.append(props[edge_color_prop])

    nx.draw_networkx_edges(
        graph,
        pos=node_positions,
        ax=axis,
        edgelist=edges,
        edge_color=edge_colors,
        edge_cmap=cmap,
        edge_vmin=0,
        edge_vmax=max_connections,
        # arrowstyle="simple",  # "simple" arrows fail with self-loops
        connectionstyle="Arc3, rad = 0.3",
        alpha=0.3,
    )

    node_labels = nx.get_node_attributes(graph, node_color_prop)
    node_labels = {key: int(val) for key, val in node_labels.items()}
    nx.draw_networkx_labels(
        graph, pos=node_positions, ax=axis, labels=node_labels, font_size=8, alpha=0.5
    )

    edge_labels = nx.get_edge_attributes(graph, edge_color_prop)
    nx.draw_networkx_edge_labels(
        graph,
        pos=node_positions,
        ax=axis,
        edge_labels=edge_labels,
        bbox={
            "boxstyle": "circle",
            "alpha": 0.2,
            "linewidth": 0,
            "facecolor": "yellowgreen",
        },
        label_pos=0.25,  # (0=head, 0.5=center, 1=tail)
        font_size=6,
        alpha=0.5,
    )

    # plt.legend()
    if show:
        plt.show(block=block)

    return axis


def plot_observation(
    host_properties,
    connections,
    axis=None,
    node_positions=None,
    show=False,
    block=False,
):
    graph = nx.DiGraph()

    for node, props in host_properties.items():
        graph.add_node(node, **props._asdict())

    for (source, target), num_connections in connections.items():
        graph.add_edge(source, target, connections=num_connections)

    axis = _plot_observation(graph, node_positions, axis=axis, show=show, block=block)
    axis.set_title("Partial observation", y=-0.1)


def plot_observation_encoded(
    env,
    state,  # encoded graph as a Data() object from pytorch geometric
    axis=None,
    node_positions=None,
    show=False,
    block=False,
):
    graph = nx.DiGraph()

    for host in env.host_names:
        host_idx = env.host_enumeration[host]
        props_vals = state.x[host_idx, :].tolist()
        # turn into named tuple and then into dict
        props_dict = env.HostProperties._make(props_vals)._asdict()
        graph.add_node(host, **props_dict)

    for (source, target), num_connections in zip(state.edge_index.T, state.edge_attr):
        source_host = env.host_enumeration.inv[int(source)]
        target_host = env.host_enumeration.inv[int(target)]
        graph.add_edge(source_host, target_host, connections=int(num_connections))

    axis = _plot_observation(graph, node_positions, axis=axis, show=show, block=block)
    axis.set_title("Encoded observation", y=-0.1)


def _plot_action_probabilities(
    action_logits, host_names, action_names, value=None, show=False, block=False
):
    probs = softmax(action_logits.flat_logits, dim=-1)
    min_prob = torch.min(probs).item()
    max_prob = torch.max(probs).item()
    global_probs = probs[:2].unsqueeze(0).detach()
    node_probs = probs[2:].reshape(action_logits.node_logits.shape).detach()

    global_action_names = action_names[:2]
    node_action_names = action_names[2:]

    fig, axs = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [action_logits.node_logits.shape[0], 1]}
    )

    node_ax = axs[0]
    global_ax = axs[1]

    mat_node = node_ax.matshow(node_probs, vmin=min_prob, vmax=max_prob)
    mat_global = global_ax.imshow(global_probs, vmin=min_prob, vmax=max_prob)

    # Show all ticks and label them with the respective list entries
    node_ax.set_yticks(np.arange(len(host_names)), labels=host_names)
    node_ax.set_xticks(np.arange(len(node_action_names)), labels=node_action_names)
    global_ax.set_yticks([0], labels=["Global"])
    global_ax.set_xticks(
        np.arange(len(global_action_names)), labels=global_action_names
    )

    if value:
        global_ax.text(
            len(node_action_names)+1,
            0,
            f"Value: {value:.2f}",
            verticalalignment="center",
            horizontalalignment="right",
            in_layout=True,
            # color="dimgrey",
            fontsize=10,
            fontweight="bold",
            bbox={
                "facecolor": "gold",
                "alpha": 0.3,
                "pad": 0.5,
                "boxstyle": "round",
            },
        )

    # Rotate the tick labels and set their alignment.
    plt.setp(node_ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    plt.setp(
        global_ax.get_xticklabels(), rotation=315, ha="left", rotation_mode="anchor"
    )

    cbar = fig.colorbar(mat_node, fraction=0.05, pad=0.05, format=StrMethodFormatter("{x:.1%}"))
    # cbar = fig.colorbar(axs, orientation="horizontal")
    cbar.set_label("Probability")

    # fig.set_tight_layout(True)
    plt.tight_layout()
    if show:
        plt.show(block=block)


def plot_action_probabilities(env, policy, obs, show=False, block=False):
    with torch.no_grad():
        action_logits, value = policy.get_action_logits(obs)
        _plot_action_probabilities(
            action_logits,
            host_names=env.host_names,
            action_names=env.action_names,
            value=value,
            show=show,
            block=block,
        )
