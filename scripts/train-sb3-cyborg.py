import inspect
from stable_baselines3 import PPO # A2C

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

max_steps = 50
agent_name = "Blue"

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
cyborg = CybORG(path, "sim", agents={'Red': RedMeanderAgent})
env = ChallengeWrapper(agent_name=agent_name, env=cyborg, max_steps=max_steps)

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=100_000, progress_bar=True)


# Example of policy inference

# num_steps = 10
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(num_steps):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     print(10*"-")