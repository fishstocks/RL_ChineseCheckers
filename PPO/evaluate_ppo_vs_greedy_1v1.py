import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO


REPO_DIR = Path(__file__).resolve().parents[1]
TEACHER_DIR = REPO_DIR / "single system"
if str(TEACHER_DIR) not in sys.path:
    sys.path.append(str(TEACHER_DIR))

from ppo_teacher_board_match import TeacherBoardMatch, greedy_action


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a 1v1 submove PPO model against the greedy baseline."
    )
    parser.add_argument(
        "--model-path",
        default=str(REPO_DIR / "ppo_1mil_1v1.zip"),
        help="Path to the submove PPO model zip.",
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--ppo-player",
        type=int,
        choices=[0, 1],
        default=0,
        help="Seat for PPO: 0 is red and moves first, 1 is blue.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default=str(REPO_DIR / "benchmark_results" / "ppo_1mil_vs_greedy_20x200.csv"),
        help="CSV path for per-game results.",
    )
    return parser.parse_args()


def run_game(model, max_steps, ppo_player):
    match = TeacherBoardMatch(max_steps=max_steps)
    obs, info = match.observe()
    done = False
    truncated = False
    extra = {}

    while not done:
        if match.current_player == ppo_player:
            action, _ = model.predict(
                obs,
                action_masks=info["action_mask"],
                deterministic=True,
            )
            action = int(action)
        else:
            action = int(greedy_action(match))

        obs, info, terminated, truncated, extra = match.step(action)
        done = terminated or truncated

    winner = extra.get("winner")
    if winner is None:
        result = "truncated"
    elif winner == ppo_player:
        result = "ppo"
    else:
        result = "greedy"

    return {
        "winner": "" if winner is None else winner,
        "result": result,
        "steps": match.step_count,
        "truncated": result == "truncated",
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    model = MaskablePPO.load(args.model_path)
    rows = []
    for game_idx in range(1, args.games + 1):
        row = run_game(
            model=model,
            max_steps=args.max_steps,
            ppo_player=args.ppo_player,
        )
        row = {
            "game": game_idx,
            "model_path": args.model_path,
            "ppo_player": args.ppo_player,
            "opponent": "greedy",
            "max_steps": args.max_steps,
            **row,
        }
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ppo_wins = sum(row["result"] == "ppo" for row in rows)
    greedy_wins = sum(row["result"] == "greedy" for row in rows)
    truncations = sum(row["result"] == "truncated" for row in rows)
    steps = [row["steps"] for row in rows]

    print(f"Saved per-game results to {output_path}")
    print(f"Games: {args.games}")
    print(f"PPO wins: {ppo_wins}")
    print(f"Greedy wins: {greedy_wins}")
    print(f"Truncations: {truncations}")
    print(f"Average steps: {np.mean(steps):.2f}")
    print(f"Min/Max steps: {min(steps)} / {max(steps)}")


if __name__ == "__main__":
    main()
