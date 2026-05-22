import os
import sys
import time
from collections import Counter
from functools import lru_cache

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from APO_env_marl_shared import ChineseCheckersEnvMARL


def _goal_centroid(env: ChineseCheckersEnvMARL, player: int):
    target_colour = env.board.colour_opposites[env.player_colours[player]]
    target_indices = env.board.axial_of_colour(target_colour)
    target_cells = [env.board.cells[i] for i in target_indices]
    goal_q = sum(c.q for c in target_cells) / len(target_cells)
    goal_r = sum(c.r for c in target_cells) / len(target_cells)
    return goal_q, goal_r, set(target_indices)


def _distance_to_goal(env: ChineseCheckersEnvMARL, idx: int, goal_q: float, goal_r: float):
    cell = env.board.cells[idx]
    return (
        abs(cell.q - goal_q)
        + abs(cell.q + cell.r - goal_q - goal_r)
        + abs(cell.r - goal_r)
    ) / 2


def _hex_distance(env: ChineseCheckersEnvMARL, src_idx: int, dst_idx: int):
    src = env.board.cells[src_idx]
    dst = env.board.cells[dst_idx]
    return (
        abs(src.q - dst.q)
        + abs((src.q + src.r) - (dst.q + dst.r))
        + abs(src.r - dst.r)
    ) / 2


def _assignment_cost(env: ChineseCheckersEnvMARL, positions, target_indices):
    positions = tuple(sorted(positions))
    targets = tuple(sorted(target_indices))

    @lru_cache(maxsize=None)
    def dp(i, used_mask):
        if i == len(positions):
            return 0.0

        best = float("inf")
        for target_i, target_idx in enumerate(targets):
            if used_mask & (1 << target_i):
                continue
            dist = _hex_distance(env, positions[i], target_idx)
            best = min(best, dist + dp(i + 1, used_mask | (1 << target_i)))
        return best

    return dp(0, 0)


def greedy_action(env: ChineseCheckersEnvMARL) -> int:
    mask = env.action_masks()
    legal_actions = np.flatnonzero(mask)
    if len(legal_actions) == 0:
        return env.END_TURN_ACTION

    player = env.current_player
    goal_q, goal_r, target_index_set = _goal_centroid(env, player)
    current_positions = [pin.axialindex for pin in env.pins[player]]
    current_assignment_cost = _assignment_cost(env, current_positions, target_index_set)

    best_score = -float("inf")
    best_actions = []

    for action in legal_actions:
        action = int(action)
        if action == env.END_TURN_ACTION:
            score = -0.25
        else:
            pin_id, direction_idx, is_jump = env._decode_action(action)
            pin = env.pins[player][pin_id]
            src_idx = pin.axialindex
            dest_idx = env._destination_for_submove(src_idx, direction_idx, is_jump)
            if dest_idx is None:
                continue

            src_dist = _distance_to_goal(env, src_idx, goal_q, goal_r)
            dest_dist = _distance_to_goal(env, dest_idx, goal_q, goal_r)
            progress = src_dist - dest_dist

            next_positions = list(current_positions)
            next_positions[pin_id] = dest_idx
            assignment_gain = current_assignment_cost - _assignment_cost(
                env, next_positions, target_index_set
            )

            score = progress + assignment_gain * 0.8
            if src_idx not in target_index_set and dest_idx in target_index_set:
                score += 2.0
            elif src_idx in target_index_set and dest_idx not in target_index_set:
                score -= 0.5
            if is_jump:
                score += 0.15

        if score > best_score + 1e-9:
            best_score = score
            best_actions = [action]
        elif abs(score - best_score) <= 1e-9:
            best_actions.append(action)

    if not best_actions:
        return env.END_TURN_ACTION
    return int(np.random.choice(best_actions))


def _clear_screen():
    os.system("clear")


def run_greedy_self_play(
    num_games=20,
    num_players=4,
    max_steps=400,
    render_last=False,
    render_live=False,
    delay=0.2,
):
    winner_counts = Counter()
    truncations = 0
    step_counts = []

    for game_idx in range(num_games):
        env = ChineseCheckersEnvMARL(
            num_players=num_players,
            render_mode="ansi" if render_last and game_idx == num_games - 1 else None,
            max_steps=max_steps,
        )
        _, info = env.reset()

        done = False
        truncated = False
        while not done:
            if render_live and game_idx == num_games - 1:
                _clear_screen()
                print(
                    f"Greedy self-play | game {game_idx + 1}/{num_games} | "
                    f"step {env.step_count} | player {env.current_player} "
                    f"({env.player_colours[env.current_player]})"
                )
                env.render()
                time.sleep(delay)
            action = greedy_action(env)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        step_counts.append(env.step_count)
        if terminated and "winner" in info:
            winner_counts[info["winner"]] += 1
        else:
            truncations += 1

        if render_last and game_idx == num_games - 1:
            print(f"\nLast greedy self-play game finished after {env.step_count} agent-turns")
            env.render()
            if "winner" in info:
                print(f"Winner: player {info['winner']} ({env.player_colours[info['winner']]})")
            elif truncated:
                print("Game ended because max steps were reached.")

    avg_steps = float(np.mean(step_counts)) if step_counts else 0.0
    print(f"\nGreedy self-play summary over {num_games} games")
    print(f"Players: {num_players}")
    print(f"Average agent-turns: {avg_steps:.2f}")
    print(f"Min/Max agent-turns: {min(step_counts)} / {max(step_counts)}")
    print(f"Truncations: {truncations}")
    for player in range(num_players):
        print(f"Player {player} wins: {winner_counts[player]}")


if __name__ == "__main__":
    num_games = int(os.environ.get("APO_BENCH_GAMES", "20"))
    num_players = int(os.environ.get("APO_NUM_PLAYERS", "4"))
    max_steps = int(os.environ.get("APO_MAX_STEPS", "400"))
    render_last = os.environ.get("APO_RENDER_LAST", "0") == "1"
    render_live = os.environ.get("APO_RENDER_LIVE", "0") == "1"
    delay = float(os.environ.get("APO_RENDER_DELAY", "0.2"))
    run_greedy_self_play(
        num_games=num_games,
        num_players=num_players,
        max_steps=max_steps,
        render_last=render_last,
        render_live=render_live,
        delay=delay,
    )
