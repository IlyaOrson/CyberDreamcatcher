from rich.pretty import pprint
from tqdm import trange
import matplotlib.pyplot as plt

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police
from cyberdreamcatcher.plots import plot_action_probabilities

# plt.show(block=False)

scenario = "Scenario2_-_User2_User4"
# scenario = "Scenario2_+_User5_User6"
env = GraphWrapper(scenario=scenario, verbose=True)

print("Observation:")
pprint(env.get_raw_observation())

print(env.get_true_table())
print(env.get_blue_table())

print("Voila!")

obs, info = env.reset()
env.render()

policy = Police(env, latent_node_dim=env.host_embedding_size)

for step in trange(3):

    action, log_prob, entropy, value = policy(obs)

    # visualise action probability distribution
    plot_action_probabilities(env, policy, obs)

    print(f"host_name = {env.host_names[action[0]]}")
    print(f"action_name = {env.action_names[action[1]]}")

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    print(env.get_true_table())
    print(env.get_blue_table())

plt.show()
