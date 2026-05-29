import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rlchinesecheckers-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rlchinesecheckers-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO


REPO_DIR = Path(__file__).resolve().parents[1]
TEACHER_DIR = REPO_DIR / "single system"
if str(TEACHER_DIR) not in sys.path:
    sys.path.append(str(TEACHER_DIR))

from ppo_teacher_board_match import TeacherBoardMatch, greedy_action


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create PPO-vs-greedy benchmark plots for the 1v1 submove model."
    )
    parser.add_argument("--model-path", default=str(REPO_DIR / "ppo_1mil_1v1.zip"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--ppo-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_DIR / "benchmark_results" / "figures"),
    )
    return parser.parse_args()


def goal_distance_sum(match, player):
    target_colour = match.board.colour_opposites[match.COLOURS[player]] if hasattr(match, "COLOURS") else None
    if target_colour is None:
        from ppo_teacher_board_match import COLOURS

        target_colour = match.board.colour_opposites[COLOURS[player]]
    target_indices = match.board.axial_of_colour(target_colour)
    target_cells = [match.board.cells[i] for i in target_indices]
    goal_q = sum(cell.q for cell in target_cells) / len(target_cells)
    goal_r = sum(cell.r for cell in target_cells) / len(target_cells)

    total = 0.0
    for pin in match.pins[player]:
        cell = match.board.cells[pin.axialindex]
        total += (
            abs(cell.q - goal_q)
            + abs(cell.q + cell.r - goal_q - goal_r)
            + abs(cell.r - goal_r)
        ) / 2
    return total


def record_heatmap_counts(match, player, counts):
    for pin in match.pins[player]:
        counts[pin.axialindex] += 1


def run_benchmark(model, games, max_steps, ppo_player):
    rows = []
    heat_counts = Counter()
    sample_trace = None
    sample_match = None

    for game_idx in range(1, games + 1):
        match = TeacherBoardMatch(max_steps=max_steps)
        obs, info = match.observe()
        done = False
        truncated = False
        extra = {}
        trace = {
            "step": [0],
            "ppo_distance": [goal_distance_sum(match, ppo_player)],
            "greedy_distance": [goal_distance_sum(match, 1 - ppo_player)],
        }
        record_heatmap_counts(match, ppo_player, heat_counts)

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

            trace["step"].append(match.step_count)
            trace["ppo_distance"].append(goal_distance_sum(match, ppo_player))
            trace["greedy_distance"].append(goal_distance_sum(match, 1 - ppo_player))
            record_heatmap_counts(match, ppo_player, heat_counts)

        winner = extra.get("winner")
        if winner is None:
            result = "truncated"
        elif winner == ppo_player:
            result = "ppo"
        else:
            result = "greedy"

        rows.append({"game": game_idx, "result": result, "steps": match.step_count})
        if sample_trace is None and result == "ppo":
            sample_trace = trace
            sample_match = match

    if sample_trace is None:
        sample_trace = trace
        sample_match = match
    return rows, heat_counts, sample_trace, sample_match


def plot_summary(rows, output_dir):
    result_counts = Counter(row["result"] for row in rows)
    labels = ["PPO wins", "Greedy wins", "Truncated"]
    values = [
        result_counts["ppo"],
        result_counts["greedy"],
        result_counts["truncated"],
    ]
    colors = ["#2f7f67", "#b85c38", "#777777"]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Games")
    ax.set_title("PPO vs Greedy Results")
    ax.set_ylim(0, max(values + [1]) + 2)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            str(value),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    path = output_dir / "ppo_vs_greedy_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_distance_trace(trace, output_dir):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(trace["step"], trace["ppo_distance"], label="PPO pieces", color="#2f7f67")
    ax.plot(
        trace["step"],
        trace["greedy_distance"],
        label="Greedy pieces",
        color="#b85c38",
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Total distance to target")
    ax.set_title("Progress Toward Target in One Game")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "ppo_vs_greedy_distance_trace.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_heatmap(match, heat_counts, output_dir):
    xs = [xy[0] for xy in match.board.cartesian]
    ys = [-xy[1] for xy in match.board.cartesian]
    counts = np.array([heat_counts[i] for i in range(len(match.board.cells))], dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 6.2))
    ax.scatter(xs, ys, s=18, color="#d5d5d5", alpha=0.7, label="Board cells")

    hot = counts > 0
    sizes = 35 + counts[hot] / max(counts.max(), 1.0) * 260
    scatter = ax.scatter(
        np.array(xs)[hot],
        np.array(ys)[hot],
        c=counts[hot],
        s=sizes,
        cmap="Reds",
        alpha=0.78,
        edgecolors="none",
    )
    fig.colorbar(scatter, ax=ax, shrink=0.78, label="PPO piece visits")
    ax.set_title("PPO Piece Position Heatmap")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    path = output_dir / "ppo_piece_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO.load(args.model_path)
    rows, heat_counts, sample_trace, sample_match = run_benchmark(
        model=model,
        games=args.games,
        max_steps=args.max_steps,
        ppo_player=args.ppo_player,
    )

    paths = [
        plot_summary(rows, output_dir),
        plot_distance_trace(sample_trace, output_dir),
        plot_heatmap(sample_match, heat_counts, output_dir),
    ]

    result_counts = Counter(row["result"] for row in rows)
    print(f"Games: {args.games}")
    print(f"PPO wins: {result_counts['ppo']}")
    print(f"Greedy wins: {result_counts['greedy']}")
    print(f"Truncations: {result_counts['truncated']}")
    print(f"Average steps: {np.mean([row['steps'] for row in rows]):.2f}")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
