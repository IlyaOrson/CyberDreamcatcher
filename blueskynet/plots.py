import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_feasible_connections(env, multipartite=False):

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
    subnet_color_map = {subnet: colors[idx] for idx, subnet in enumerate(env.subnet_names)}

    for subnet, hostnames in env.subnet_hostnames_map.items():
        graph.add_nodes_from(hostnames, subnet=subnet)
    graph.add_edges_from(env.feasible_connections)

    if multipartite:
        positions = nx.multipartite_layout(graph, subset_key="subnet", align="horizontal")
    else:
        positions = nx.kamada_kawai_layout(graph)

    for subnet, hostnames in env.subnet_hostnames_map.items():

        color = subnet_color_map[subnet]
        node_colors = len(hostnames) * [color]

        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=hostnames,
            node_color=node_colors,
            # node_size=1000,
            label=subnet,
            # cmap=cmap
        )

    nx.draw_networkx_edges(graph, pos=positions, alpha=0.4)
    nx.draw_networkx_labels(graph, pos=positions, font_size=8, alpha=0.8)

    plt.legend()
    plt.show()
