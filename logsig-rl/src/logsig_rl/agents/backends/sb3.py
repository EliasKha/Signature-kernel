from gymnasium import Env
from stable_baselines3 import SAC, TD3, DDPG, PPO, A2C

def build_sb3_model(algo: str, env: Env):
    algo = algo.lower()
    if algo == "sac":
        return SAC("MlpPolicy", env, verbose=0, device="auto")
    if algo == "td3":
        return TD3("MlpPolicy", env, verbose=0, device="auto")
    if algo == "ddpg":
        return DDPG("MlpPolicy", env, verbose=0, device="auto")
    if algo == "ppo":
        return PPO("MlpPolicy", env, verbose=0, device="auto")
    if algo == "a2c":
        return A2C("MlpPolicy", env, verbose=0, device="auto")
    raise ValueError(f"Unknown algo {algo}")
