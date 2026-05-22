import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO


ROOT_DIR = Path(__file__).resolve().parents[1]
PPO_DIR = ROOT_DIR / "PPO"
if str(PPO_DIR) not in sys.path:
    sys.path.append(str(PPO_DIR))

from APO_env_marl_shared import ChineseCheckersEnvMARL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch a 4-player PPO model play against three opponent PPO models."
    )
    parser.add_argument("--model-path", required=True, help="Path to the 4-player PPO zip.")
    parser.add_argument(
        "--opponent-model",
        required=True,
        help="Path to the PPO zip used for the other three seats.",
    )
    parser.add_argument(
        "--seat",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Seat index (0-3) for the main 4-player model.",
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.2)
    return parser.parse_args()


def render_env(env, seat_for_main_model):
    print("\033[2J\033[H", end="", flush=True)
    print(
        f"\nStep {env.step_count} | player {env.current_player} "
        f"({env.player_colours[env.current_player]})"
    )
    print(f"Main 4p model seat: {seat_for_main_model} ({env.player_colours[seat_for_main_model]})")
    all_pins = []
    for player_pins in env.pins:
        all_pins.extend(player_pins)
    env.board.print_ascii(pins=all_pins, empty="·")


def main():
    args = parse_args()

    main_model = MaskablePPO.load(args.model_path)
    opponent_model = MaskablePPO.load(args.opponent_model)

    env = ChineseCheckersEnvMARL(num_players=4, max_steps=args.max_steps)
    obs, info = env.reset()

    print(f"Loaded 4-player PPO: {Path(args.model_path).name}")
    print(f"Loaded opponent PPO: {Path(args.opponent_model).name}")
    print(f"Main model seat: {args.seat}")

    done = False
    truncated = False
    extra = {}

    if args.render:
        render_env(env, args.seat)

    while not done:
        current = info["current_player"]
        model = main_model if current == args.seat else opponent_model
        legal_mask = np.asarray(info["action_mask"], dtype=bool)
        action, _ = model.predict(
            obs, action_masks=info["action_mask"], deterministic=True
        )
        action = int(action)

        if action < 0 or action >= len(legal_mask) or not legal_mask[action]:
            legal_actions = np.flatnonzero(legal_mask)
            if len(legal_actions) == 0:
                raise RuntimeError("No legal actions available for fallback.")
            action = int(legal_actions[0])

        obs, _, terminated, truncated, extra = env.step(action)

        if extra.get("illegal"):
            legal_actions = np.flatnonzero(legal_mask)
            if len(legal_actions) == 0:
                raise RuntimeError("Model produced illegal action and no legal fallback exists.")
            obs, _, terminated, truncated, extra = env.step(int(legal_actions[0]))

        done = terminated or truncated

        if args.render:
            render_env(env, args.seat)

        if args.delay > 0 and not done:
            time.sleep(args.delay)

    print(f"\nFinished after {env.step_count} steps")
    if "winner" in extra:
        winner = extra["winner"]
        print(f"Winner: player {winner} ({env.player_colours[winner]})")
    elif truncated:
        print("Game ended by max steps.")


if __name__ == "__main__":
    main()
