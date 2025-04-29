from rich import inspect
from rich.pretty import pprint
from tqdm import trange
import matplotlib.pyplot as plt
import torch

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police
from cyberdreamcatcher.plots import plot_action_probabilities, plot_observation_encoded

# plt.show(block=False)

scenario = None
# scenario = "Scenario2"
# scenario = "Scenario2_-_User2_User4"
# scenario = "Scenario2_+_User5_User6"
env = GraphWrapper(scenario=scenario, verbose=True)

obs, info = env.reset()
env.render()

pprint("---Initial Observation---")
pprint(env.get_raw_observation())

pprint("---True Table---")
pprint(env.get_true_table())
pprint("---Blue Table---")
pprint(env.get_blue_table())


policy = Police(env)

for step in trange(30):

    action = torch.tensor(env.action_space.sample())
    # action, log_prob, entropy, value = policy(obs)

    # visualise action probability distribution
    # plot_action_probabilities(env, policy, obs)

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    pprint(f"---Step {step}---")
    pprint("---Action---")
    inspect(info["action"])
    pprint("---Observation---")
    pprint(info["observation"])

    pprint("---Hosts---")
    pprint(info["hosts"])

    pprint("---Connections---")
    pprint(info["connections"])

    # plot_observation_encoded(env, obs, show=True)

    pprint("---Last Red Action---")
    inspect(env.cyborg.get_last_action(agent="Red"))
    pprint("---Last Blue Action---")
    inspect(env.cyborg.get_last_action(agent="Blue"))

    pprint("---True Table---")
    pprint(env.get_true_table())
    pprint("---Blue Table---")
    pprint(env.get_blue_table())

plt.show()
