import os
import sys
import time

from sb3_contrib import MaskablePPO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from APO_env_marl import ChineseCheckersEnvMARL


def _find_model_path(num_players: int) -> str:
    env_override = os.environ.get("APO_MODEL_PATH")
    if env_override:
        if not os.path.exists(env_override):
            raise FileNotFoundError(f"APO_MODEL_PATH does not exist: {env_override}")
        return env_override

    candidate_paths = [
        f"checkers_ppo_marl_{num_players}p.zip",
        "checkers_ppo_marl.zip",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path

    checkpoint_dir = "checkpoints"
    prefix = "checkers_ppo_marl"
    if os.path.isdir(checkpoint_dir):
        files = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith(prefix) and f.endswith(".zip")
        ]
        if files:
            latest = max(files, key=lambda f: int(f.split("_steps")[0].split("_")[-1]))
            return os.path.join(checkpoint_dir, latest)

    raise FileNotFoundError(
        "No MARL model found. Train first with APO_train_marl.py or set APO_MODEL_PATH."
    )


def main():
    num_players = int(os.environ.get("APO_NUM_PLAYERS", "4"))
    delay_seconds = float(os.environ.get("APO_EVAL_DELAY", "0.3"))
    deterministic = os.environ.get("APO_DETERMINISTIC", "1") != "0"

    model_path = _find_model_path(num_players)
    env = ChineseCheckersEnvMARL(num_players=num_players, render_mode="ansi")
    model = MaskablePPO.load(model_path, env=env)

    obs, info = env.reset()
    done = False
    step = 0

    print(f"Watching one MARL-style shared-policy game with {num_players} players...")
    print(f"Loaded model: {model_path}")
    time.sleep(1.0)

    while not done:
        os.system("clear")
        print(
            f"Step {step} | Current player: {info['current_player']} "
            f"({info['current_colour']}) | Players: {info['num_players']}"
        )
        env.render()
        action, _ = model.predict(
            obs,
            action_masks=info["action_mask"],
            deterministic=deterministic,
        )
        print(f"Chosen action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}")
        done = terminated or truncated
        step += 1
        time.sleep(delay_seconds)

    os.system("clear")
    print(f"\nGame finished after {step} steps")
    env.render()

    if "winner" in info:
        winner = info["winner"]
        print(f"Winner: player {winner} ({env.player_colours[winner]})")
    elif truncated:
        print("Game ended because max steps were reached.")
    else:
        print("Game ended without a recorded winner.")


if __name__ == "__main__":
    main()
