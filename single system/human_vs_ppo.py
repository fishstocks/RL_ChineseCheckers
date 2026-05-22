import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rlchinesecheckers-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rlchinesecheckers-cache")

from sb3_contrib import MaskablePPO

from ppo_teacher_board_match import (
    COLOURS,
    END_TURN_ACTION,
    TeacherBoardMatch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Play locally as a human against a PPO model.")
    parser.add_argument(
        "--model-path",
        default=str(Path(__file__).resolve().parents[1] / "ppo_1mil_1v1.zip"),
        help="Path to the PPO zip model.",
    )
    parser.add_argument(
        "--human-player",
        type=int,
        choices=[0, 1],
        default=1,
        help="Human seat: 0 is red and moves first, 1 is blue and moves second.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--ppo-delay", type=float, default=0.3)
    return parser.parse_args()


def legal_action_rows(match):
    rows = []
    for action in np.flatnonzero(match.action_masks()):
        action = int(action)
        if action == END_TURN_ACTION:
            rows.append((action, "end turn", "", ""))
            continue

        pin_id, direction_idx, is_jump = match._decode_action(action)
        pin = match.pins[match.current_player][pin_id]
        dest_idx = match._destination_for_submove(pin.axialindex, direction_idx, is_jump)
        move_type = "jump" if is_jump else "step"
        rows.append((action, f"pin {pin_id}", f"{pin.axialindex} -> {dest_idx}", move_type))
    return rows


def print_turn(match):
    colour = COLOURS[match.current_player]
    print("\n" + "=" * 60)
    print(f"Step {match.step_count} | {colour.upper()}'s turn")
    print("=" * 60)
    match.render()
    print("\nYour pins:")
    print([(pin.id, pin.axialindex) for pin in match.pins[match.current_player]])
    print("\nLegal actions:")
    for action, pin, path, move_type in legal_action_rows(match):
        if action == END_TURN_ACTION:
            print(f"  {action:3d}: end turn")
        else:
            print(f"  {action:3d}: {pin:6s} {path:10s} {move_type}")


def action_from_human(match):
    legal = {row[0] for row in legal_action_rows(match)}
    while True:
        raw = input("\nChoose action number, or q to quit: ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            action = int(raw)
        except ValueError:
            print("Please type one of the listed action numbers.")
            continue
        if action not in legal:
            print("That action is not legal right now.")
            continue
        return action


def main():
    args = parse_args()
    model = MaskablePPO.load(args.model_path)
    match = TeacherBoardMatch(max_steps=args.max_steps)
    obs, info = match.observe()
    human_colour = COLOURS[args.human_player]
    ppo_colour = COLOURS[1 - args.human_player]

    print(f"Loaded PPO model: {args.model_path}")
    print(f"You are player {args.human_player} ({human_colour}). PPO is {ppo_colour}.")

    done = False
    truncated = False
    extra = {}
    try:
        while not done:
            if match.current_player == args.human_player:
                print_turn(match)
                action = action_from_human(match)
            else:
                action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
                action = int(action)
                print(f"\nPPO ({COLOURS[match.current_player]}) chose action {action}")
                if args.ppo_delay > 0:
                    time.sleep(args.ppo_delay)

            obs, info, terminated, truncated, extra = match.step(action)
            done = terminated or truncated
    except KeyboardInterrupt:
        print("\nStopped.")
        return

    print("\nFinal board:")
    match.render()
    if "winner" in extra:
        winner = extra["winner"]
        print(f"Winner: player {winner} ({COLOURS[winner]})")
    elif truncated:
        print("Game ended by max steps.")


if __name__ == "__main__":
    main()
