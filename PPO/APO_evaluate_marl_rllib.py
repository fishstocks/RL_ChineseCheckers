import os
import sys
import time

import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core import Columns
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from APO_env_marl import env as make_env


def _find_checkpoint(num_players: int, policy_mode: str) -> str:
    env_override = os.environ.get("APO_MODEL_PATH")
    if env_override:
        if not os.path.exists(env_override):
            raise FileNotFoundError(f"APO_MODEL_PATH does not exist: {env_override}")
        return os.path.abspath(env_override)

    candidate = os.path.abspath(
        os.path.join("checkpoints", f"rllib_marl_{num_players}p_{policy_mode}")
    )
    if os.path.exists(candidate):
        return candidate

    checkpoint_root = os.path.abspath("checkpoints")
    prefix = f"rllib_marl_{num_players}p_"
    if os.path.isdir(checkpoint_root):
        candidates = [
            os.path.join(checkpoint_root, name)
            for name in os.listdir(checkpoint_root)
            if name.startswith(prefix)
        ]
        if candidates:
            return max(candidates, key=os.path.getmtime)

    raise FileNotFoundError(
        "No RLlib MARL checkpoint found. Train first with APO_train_marl.py "
        "or set APO_MODEL_PATH."
    )


def _infer_policy_mode(checkpoint_path: str) -> str:
    basename = os.path.basename(os.path.normpath(checkpoint_path)).lower()
    if "independent" in basename:
        return "independent"
    return os.environ.get("APO_POLICY_MODE", "shared").strip().lower()


def _policy_id_for_agent(agent_id: str, policy_mode: str) -> str:
    if policy_mode == "shared":
        return "shared_policy"
    return agent_id


def _register_env(num_players: int, max_steps: int) -> None:
    env_name = f"apo_chinese_checkers_marl_{num_players}p"

    def env_creator(_config):
        aec_env = make_env(num_players=num_players, max_steps=max_steps, render_mode="ansi")
        return PettingZooEnv(aec_env)

    register_env(env_name, env_creator)


def _clear_screen():
    os.system("clear")


def _compute_action(algo: Algorithm, policy_id: str, obs: dict, deterministic: bool):
    module = algo.get_module(policy_id)
    batch = {
        Columns.OBS: {
            "observation": torch.as_tensor(
                obs["observation"], dtype=torch.float32
            ).unsqueeze(0),
            "action_mask": torch.as_tensor(
                obs["action_mask"], dtype=torch.float32
            ).unsqueeze(0),
        }
    }
    outputs = module.forward_inference(batch)
    dist_cls = module.get_inference_action_dist_cls()
    dist = dist_cls.from_logits(outputs[Columns.ACTION_DIST_INPUTS])
    if deterministic:
        dist = dist.to_deterministic()
    action = dist.sample()
    if isinstance(action, tuple):
        action = action[0]
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    if hasattr(action, "item"):
        return int(action.item())
    return int(action[0])


def main():
    num_players = int(os.environ.get("APO_NUM_PLAYERS", "4"))
    delay_seconds = float(os.environ.get("APO_EVAL_DELAY", "0.3"))
    deterministic = os.environ.get("APO_DETERMINISTIC", "1") != "0"
    max_steps = int(os.environ.get("APO_MAX_STEPS", "400"))

    checkpoint_path = _find_checkpoint(
        num_players=num_players,
        policy_mode=os.environ.get("APO_POLICY_MODE", "shared").strip().lower(),
    )
    policy_mode = _infer_policy_mode(checkpoint_path)
    _register_env(num_players=num_players, max_steps=max_steps)

    algo = Algorithm.from_checkpoint(checkpoint_path)
    env = make_env(num_players=num_players, max_steps=max_steps, render_mode="ansi")
    env.reset()

    step = 0
    print(
        f"Watching one RLlib MARL game | players={num_players} | policy_mode={policy_mode}"
    )
    print(f"Loaded checkpoint: {checkpoint_path}")
    time.sleep(1.0)

    while env.agents:
        agent_id = env.agent_selection
        obs, reward, terminated, truncated, info = env.last()

        _clear_screen()
        player_idx = env.unwrapped.agent_name_mapping[agent_id]
        current_colour = env.unwrapped.player_colours[player_idx]
        print(
            f"Step {step} | Agent: {agent_id} | Player: {player_idx} ({current_colour}) "
            f"| Players: {num_players}"
        )
        env.render()
        print(f"Last reward for {agent_id}: {reward:.3f}")

        if terminated or truncated:
            print("Agent is done; advancing with null action.")
            action = None
        else:
            action = _compute_action(
                algo=algo,
                policy_id=_policy_id_for_agent(agent_id, policy_mode),
                obs=obs,
                deterministic=deterministic,
            )
            print(f"Chosen action: {action}")

        env.step(action)
        step += 1
        time.sleep(delay_seconds)

    _clear_screen()
    print(f"Game finished after {step} agent-turns")
    env.render()

    winner = env.unwrapped.winner
    if winner is not None:
        print(f"Winner: player {winner} ({env.unwrapped.player_colours[winner]})")
    elif any(env.unwrapped.truncations.values()):
        print("Game ended because max steps were reached.")
    else:
        print("Game ended without a recorded winner.")

    algo.stop()
    env.close()


if __name__ == "__main__":
    main()
