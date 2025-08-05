"""
logsig_rl – A tiny research package for reinforcement-learning with
log-signature features and a signature-MPC controller.
Expose the public API so users can just do:

    from logsig_rl import SignatureAgent, SignatureMPCAgent, SigPrefixEnv
"""
from .agents.rl import GenericRLAgent
from .agents.mpc import SignatureMPCAgent
from .envs.sig_prefix_env import SigPrefixEnv

__all__ = [
    "GenericRLAgent",
    "SignatureMPCAgent",
    "SigPrefixEnv",
]
