from rich import inspect
from rich.console import Console
from rich.rule import Rule
from tqdm import trange
import matplotlib.pyplot as plt
import torch

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police
# from cyberdreamcatcher.plots import plot_action_probabilities, plot_observation_encoded

console = Console()

# plt.show(block=False)

scenario = None
# scenario = "Scenario2"
# scenario = "Scenario2_-_User2_User4"
# scenario = "Scenario2_+_User5_User6"
env = GraphWrapper(scenario=scenario, verbose=True, track_red_table=True)

obs, info = env.reset()
# env.render()

console.print(Rule("Initial Observation", style="bold red"))
# pprint(info["observation"])

console.print(Rule("True Table", style="bold red"))
console.print(info["true_table"])
console.print(Rule("Blue Table", style="bold red"))
console.print(info["blue_table"])
if env.red_table:
    console.print(Rule("Red Table", style="bold red"))
    console.print(info["red_table"])

policy = Police(env)

for step in trange(30):

    action = torch.tensor(env.action_space.sample())
    # action, log_prob, entropy, value = policy(obs)

    # visualise action probability distribution
    # plot_action_probabilities(env, policy, obs)
    
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render()
    # plot_observation_encoded(env, obs, show=True)

    console.print(Rule(f"Step {step}", style="bold red"))
    console.print(Rule("Action", style="bold red"))
    inspect(info["action"])
    console.print(Rule("Observation", style="bold red"))
    console.print(info["observation"])

    console.print(Rule("Hosts", style="bold red"))
    console.print(info["hosts"])
    console.print(Rule("Connections", style="bold red"))
    console.print(info["connections"])

    console.print(Rule("True Table", style="bold red"))
    console.print(info["true_table"])

    console.print(Rule("Last Blue Action", style="bold red"))
    inspect(env.cyborg.get_last_action(agent="Blue"))
    console.print(Rule("Blue Table", style="bold red"))
    console.print(info["blue_table"])

    console.print(Rule("Last Red Action", style="bold red"))
    inspect(env.cyborg.get_last_action(agent="Red"))

    if env.red_table:
        console.print(Rule("Red Table", style="bold red"))
        console.print(info["red_table"])
        console.print(Rule("Red Observation", style="bold red"))
        console.print(info["red_obs"])
    
    console.print(Rule("Reward", style="bold red"))
    console.print(env.cyborg.get_rewards())

plt.show()
