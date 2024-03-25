from rich.pretty import pprint
import networkx as nx
import matplotlib.pyplot as plt

from blueskynet.env import GraphWrapper

env = GraphWrapper()
print("Observation:")
pprint(env.get_observation())
print(env.get_table())


# TODO setup as multipartite graph
# https://networkx.org/documentation/stable/auto_examples/drawing/plot_multipartite_graph.html#multipartite-layout

graph = nx.DiGraph()
graph.add_nodes_from(env.host_names)
graph.add_edges_from(env.feasible_connections)
positions = nx.kamada_kawai_layout(graph)
nx.draw_networkx(graph, pos=positions)
plt.show()
print("Voila!")

# FIXME BaseWrapper.get_action_space calls the wrapped env method
# action_space = env.get_action_space("Blue")

# results = env.reset()
# agent = TestAgent()  # selects random actions

# for step in range(5):
#     action = agent.get_action(results.observation, results.action_space)
#     results = env.step(action=action, agent='Red')
#     print(results.reward)


# TODO adapt to sb3 api
# from stable_baselines3 import PPO
# model = PPO("MultiInputPolicy", env, verbose=1, device="cpu")

# num_steps = 10
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(num_steps):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     print(10*"-")
