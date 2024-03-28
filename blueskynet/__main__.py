from rich.pretty import pprint

# from CybORG.Agents import TestAgent
from blueskynet.env import GraphWrapper
from blueskynet.plots import plot_feasible_connections

# scenario = None
scenario = "Scenario2_+_user5"
env = GraphWrapper(scenario=scenario)

print("Observation:")
pprint(env.get_observation())

print(env.get_true_table())
print(env.get_blue_table())

plot_feasible_connections(env)

print("Voila!")

obs, info = env.reset()

for step in range(5):
    num_actions = info["action_space"]
    # TestAgent selects random actions from the CybORG level action space
    # action = TestAgent().get_action(obs, env.cyborg.get_action_space(env.agent_name))
    action = env.env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)

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
