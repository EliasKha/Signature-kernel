import gymnasium as gym
import numpy as np

class FeatureAdapterObsWrapper(gym.ObservationWrapper):
    """Wraps an env and replaces obs with adapter.transform(obs)."""
    def __init__(self, env: gym.Env, adapter):
        super().__init__(env)
        self.adapter = adapter
        self.adapter.reset()

        # Probe to set observation_space
        obs, _ = self.env.reset()
        z = self.adapter.transform(obs)
        z = np.asarray(z, dtype=np.float32).ravel()
        low = np.full_like(z, -np.inf, dtype=np.float32)
        high = np.full_like(z, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        z = self.adapter.transform(observation)
        return np.asarray(z, dtype=np.float32).ravel()

    def reset(self, **kwargs):
        self.adapter.reset()
        return super().reset(**kwargs)
