from stable_baselines3 import PPO # A2C

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

from blueskynet.utils import get_scenario

max_steps = 30
agent_name = "Blue"

# scenario_path = get_scenario(name="Scenario2", from_cyborg=True)
scenario_path = get_scenario(name="Scenario2_+_user5", from_cyborg=False)
cyborg = CybORG(scenario_path, "sim", agents={'Red': RedMeanderAgent})
env = ChallengeWrapper(agent_name=agent_name, env=cyborg, max_steps=max_steps)

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=1_000_000, progress_bar=True)  # 1_000_000 produces competitive results


# Example of policy inference

# num_steps = 10
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(num_steps):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     print(10*"-")
