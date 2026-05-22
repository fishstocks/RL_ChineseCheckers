import os

import ray.rllib.algorithms.algorithm as rllib_algorithm
import ray.train.constants as ray_train_constants
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from APO_action_mask_rllib import TorchActionMaskRLM
from APO_env_marl import env as make_env


NUM_PLAYERS = int(os.environ.get("APO_NUM_PLAYERS", "4"))
POLICY_MODE = os.environ.get("APO_POLICY_MODE", "shared").strip().lower()
MAX_STEPS = int(os.environ.get("APO_MAX_STEPS", "400"))
TRAINING_ITERS = int(os.environ.get("APO_TRAINING_ITERS", "100"))
TIMESTEPS_PER_ITER = int(os.environ.get("APO_TIMESTEPS_PER_ITER", "4000"))
RAY_RESULTS_DIR = os.path.abspath("ray_results")
os.makedirs(RAY_RESULTS_DIR, exist_ok=True)
os.environ.setdefault("RAY_AIR_LOCAL_CACHE_DIR", RAY_RESULTS_DIR)
ray_train_constants.DEFAULT_STORAGE_PATH = RAY_RESULTS_DIR
rllib_algorithm.DEFAULT_STORAGE_PATH = RAY_RESULTS_DIR

if POLICY_MODE not in {"shared", "independent"}:
    raise ValueError("APO_POLICY_MODE must be 'shared' or 'independent'")


def env_creator(_config):
    aec_env = make_env(num_players=NUM_PLAYERS, max_steps=MAX_STEPS, render_mode="ansi")
    return PettingZooEnv(aec_env)


ENV_NAME = f"apo_chinese_checkers_marl_{NUM_PLAYERS}p"
register_env(ENV_NAME, env_creator)

temp_env = env_creator({})
possible_agents = list(temp_env.env.possible_agents)
observation_space = temp_env.env.observation_space(possible_agents[0])
action_space = temp_env.env.action_space(possible_agents[0])

if POLICY_MODE == "shared":
    policies = {
        "shared_policy": (
            None,
            observation_space,
            action_space,
            {},
        )
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_policy"
else:
    policies = {
        agent_id: (
            None,
            observation_space,
            action_space,
            {},
        )
        for agent_id in possible_agents
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return agent_id


config = (
    PPOConfig()
    .environment(env=ENV_NAME)
    .framework("torch")
    .env_runners(num_env_runners=0)
    .training(
        train_batch_size=TIMESTEPS_PER_ITER,
        lr=1e-4,
        clip_param=0.1,
        grad_clip=0.5,
        entropy_coeff=0.005,
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=TorchActionMaskRLM,
            observation_space=observation_space,
            action_space=action_space,
            model_config={"fcnet_hiddens": [128, 128]},
        )
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=list(policies.keys()),
    )
)

algo = config.build()

print(
    f"Starting true MARL PPO training with RLlib | "
    f"players={NUM_PLAYERS} | policy_mode={POLICY_MODE}"
)


def _metric(result, *path):
    value = result
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value

for iteration in range(1, TRAINING_ITERS + 1):
    result = algo.train()
    print(
        f"iter={iteration} "
        f"episode_return_mean={_metric(result, 'env_runners', 'episode_return_mean')} "
        f"episode_len_mean={_metric(result, 'env_runners', 'episode_len_mean')} "
        f"env_steps_sampled={result.get('num_env_steps_sampled_lifetime')}"
    )

output_dir = os.path.abspath(
    os.path.join("checkpoints", f"rllib_marl_{NUM_PLAYERS}p_{POLICY_MODE}")
)
os.makedirs(output_dir, exist_ok=True)
save_result = algo.save(checkpoint_dir=output_dir)
print(f"Saved RLlib MARL checkpoint to: {save_result.checkpoint.path}")
