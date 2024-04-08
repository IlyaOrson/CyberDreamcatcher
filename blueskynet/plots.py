import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_feasible_connections(env, axis=None, multipartite=False):
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

    nx.draw_networkx_edges(graph, pos=node_positions, ax=axis, alpha=0.4)
    nx.draw_networkx_labels(graph, pos=node_positions, ax=axis, font_size=8, alpha=0.8)

    plt.title(f"Layout from {env.scenario_name}")
    plt.legend()
    plt.show(block=False)

    return node_positions


def plot_observation(
    host_properties, connections, action_name=None, axis=None, node_positions=None
):
    # Simplest version
    # graph = nx.DiGraph()
    # graph.add_nodes_from(env.host_names)
    # graph.add_edges_from(env.feasible_connections)
    # positions = nx.kamada_kawai_layout(graph)
    # nx.draw_networkx(graph, pos=positions)
    # plt.show()

    graph = nx.DiGraph()

    for node, props in host_properties.items():
        graph.add_node(node, **props._asdict())

    for (source, target), num_connections in connections.items():
        graph.add_edge(source, target, connections=num_connections)

    if node_positions is None:
        node_positions = nx.kamada_kawai_layout(graph)

    if axis is None:
        _, axis = plt.subplots()
    else:
        axis.cla()

    nx.draw_networkx_nodes(graph, pos=node_positions, ax=axis)
    nx.draw_networkx_edges(graph, pos=node_positions, ax=axis, alpha=0.3)
    nx.draw_networkx_labels(graph, pos=node_positions, ax=axis, font_size=8, alpha=0.8)

    edge_labels = nx.get_edge_attributes(graph, "connections")
    nx.draw_networkx_edge_labels(
        graph,
        pos=node_positions,
        ax=axis,
        edge_labels=edge_labels,
        font_size=8,
        alpha=0.8,
    )

    if action_name is None:
        plt.title("Initial blue observation")
    else:
        plt.title(f"Blue observation after {action_name}")
    plt.legend()
    plt.show(block=False)
