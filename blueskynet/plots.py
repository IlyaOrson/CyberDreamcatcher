from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from torch.nn.functional import softmax


def plot_feasible_connections(
    env, axis=None, multipartite=False, show=False, block=False
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

    for subnet, hostnames in env.subnet_hostnames_map.items():
        color = subnet_color_map[subnet]
        node_colors = len(hostnames) * [color]

        nx.draw_networkx_nodes(
            graph,
            pos=node_positions,
            ax=axis,
            nodelist=hostnames,
            node_color=node_colors,
            # node_size=1000,
            label=subnet,
            # cmap=cmap
        )

    nx.draw_networkx_edges(
        graph, pos=node_positions, ax=axis, connectionstyle="Arc3, rad = 0.2", alpha=0.3
    )
    nx.draw_networkx_labels(graph, pos=node_positions, ax=axis, font_size=8, alpha=0.5)

    plt.title(f"Layout from {env.scenario_name}")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show(block=block)

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


def plot_action_probabilities(
    logits, host_names, action_names, show=False, block=False
):
    probs = softmax(logits.flatten(), dim=-1).reshape(logits.shape).detach()

    fig, ax = plt.subplots()

    mat = ax.matshow(probs)
    cbar = fig.colorbar(mat, fraction=0.05, pad=0.05)
    # cbar = fig.colorbar(mat, orientation="horizontal")
    cbar.set_label("Probability")

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(host_names)), labels=host_names)
    ax.set_xticks(np.arange(len(action_names)), labels=action_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    fig.set_tight_layout(True)
    if show:
        plt.show(block=block)
