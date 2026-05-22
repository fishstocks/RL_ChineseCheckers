import argparse
import io
import os
import sys
import time
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from checkers_board import HexBoard
from checkers_pins import Pin

MCTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MCTS"))
if MCTS_DIR not in sys.path:
    sys.path.append(MCTS_DIR)

from neural_guided_mcts import NeuralGuide, NeuralGuidedMCTS, SearchState


COLOURS = ["red", "blue"]
DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
]
N_PINS = 10
ACTIONS_PER_PIN = len(DIRECTIONS) * 2
END_TURN_ACTION = N_PINS * ACTIONS_PER_PIN
ACTION_DIM = END_TURN_ACTION + 1


class TeacherBoardMatch:
    def __init__(self, max_steps=300):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        with redirect_stdout(io.StringIO()):
            self.board = HexBoard(R=4, hole_radius=16, spacing=34)
        self.pins = [[], []]
        self.current_player = 0
        self.step_count = 0
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()
        self.done = False
        self.winner = None

        with redirect_stdout(io.StringIO()):
            for player, colour in enumerate(COLOURS):
                axials = self.board.axial_of_colour(colour)
                self.pins[player] = [
                    Pin(self.board, axials[i], id=i, color=colour)
                    for i in range(N_PINS)
                ]

        return self.observe()

    def observe(self):
        return self._get_obs(), {"action_mask": self.action_masks()}

    def _get_obs(self):
        n_cells = len(self.board.cells)
        l1 = np.zeros(n_cells, dtype=np.float32)
        for p in self.pins[self.current_player]:
            l1[p.axialindex] = 1.0

        l2 = np.zeros(n_cells, dtype=np.float32)
        for p in self.pins[1 - self.current_player]:
            l2[p.axialindex] = 1.0

        l3 = np.zeros(n_cells, dtype=np.float32)
        target_colour = self.board.colour_opposites[COLOURS[self.current_player]]
        for idx in self.board.axial_of_colour(target_colour):
            l3[idx] = 1.0

        l4 = np.zeros(n_cells, dtype=np.float32)
        if self.active_jump_pin_id is not None:
            active_pin = self.pins[self.current_player][self.active_jump_pin_id]
            l4[active_pin.axialindex] = 1.0

        return np.concatenate([l1, l2, l3, l4]).astype(np.float32)

    def action_masks(self):
        mask = np.zeros(ACTION_DIM, dtype=bool)

        if self.active_jump_pin_id is not None:
            pin = self.pins[self.current_player][self.active_jump_pin_id]
            jump_found = False
            for direction_idx in range(len(DIRECTIONS)):
                if self._is_legal_submove(pin, direction_idx, is_jump=True):
                    mask[self._encode_action(pin.id, direction_idx, True)] = True
                    jump_found = True
            mask[END_TURN_ACTION] = True
            if not jump_found:
                mask[END_TURN_ACTION] = True
            return mask

        has_any_move = False
        for pin in self.pins[self.current_player]:
            for direction_idx in range(len(DIRECTIONS)):
                if self._is_legal_submove(pin, direction_idx, is_jump=False):
                    mask[self._encode_action(pin.id, direction_idx, False)] = True
                    has_any_move = True
                if self._is_legal_submove(pin, direction_idx, is_jump=True):
                    mask[self._encode_action(pin.id, direction_idx, True)] = True
                    has_any_move = True

        if not has_any_move:
            mask[END_TURN_ACTION] = True

        return mask

    def step(self, action):
        if self.done:
            raise RuntimeError("Match is done. Call reset() first.")

        legal_mask = self.action_masks()
        action = int(action)
        if action < 0 or action >= ACTION_DIM or not legal_mask[action]:
            raise ValueError(f"Illegal action {action} for player {self.current_player}")

        if action == END_TURN_ACTION:
            self.step_count += 1
            self._switch_turn()
            return self.observe() + (False, self.step_count >= self.max_steps, {})

        pin_id, direction_idx, is_jump = self._decode_action(action)
        pin = self.pins[self.current_player][pin_id]
        src_idx = pin.axialindex
        dest_idx = self._destination_for_submove(src_idx, direction_idx, is_jump)

        with redirect_stdout(io.StringIO()):
            pin.placePin(dest_idx)

        self.step_count += 1

        if is_jump:
            if self.active_jump_pin_id is None:
                self.active_jump_pin_id = pin_id
                self.jump_visited_cells = {src_idx, dest_idx}
            else:
                self.jump_visited_cells.add(dest_idx)

        if self._has_player_won(self.current_player):
            self.done = True
            self.winner = self.current_player
            return self.observe() + (True, False, {"winner": self.winner})

        if self.step_count >= self.max_steps:
            self.done = True
            return self.observe() + (False, True, {"reason": "max_steps"})

        if not is_jump or not self._has_continuing_jump(pin_id):
            self._switch_turn()

        return self.observe() + (False, False, {})

    def render(self):
        self.board.print_ascii(pins=self.pins[0] + self.pins[1], empty="·")

    def _encode_action(self, pin_id, direction_idx, is_jump):
        return pin_id * ACTIONS_PER_PIN + direction_idx * 2 + int(is_jump)

    def _decode_action(self, action):
        pin_id = action // ACTIONS_PER_PIN
        local = action % ACTIONS_PER_PIN
        direction_idx = local // 2
        is_jump = bool(local % 2)
        return pin_id, direction_idx, is_jump

    def _destination_for_submove(self, src_idx, direction_idx, is_jump):
        src_cell = self.board.cells[src_idx]
        dq, dr = DIRECTIONS[direction_idx]
        multiplier = 2 if is_jump else 1
        q = src_cell.q + dq * multiplier
        r = src_cell.r + dr * multiplier
        return self.board.index_of.get((q, r))

    def _is_legal_submove(self, pin, direction_idx, is_jump):
        if self.active_jump_pin_id is not None and pin.id != self.active_jump_pin_id:
            return False

        src_idx = pin.axialindex
        src_cell = self.board.cells[src_idx]
        dq, dr = DIRECTIONS[direction_idx]

        if is_jump:
            adj_idx = self.board.index_of.get((src_cell.q + dq, src_cell.r + dr))
            dst_idx = self.board.index_of.get((src_cell.q + 2 * dq, src_cell.r + 2 * dr))
            if adj_idx is None or dst_idx is None:
                return False
            if not self.board.cells[adj_idx].occupied:
                return False
            if self.board.cells[dst_idx].occupied:
                return False
            if dst_idx in self.jump_visited_cells:
                return False
            return True

        if self.active_jump_pin_id is not None:
            return False

        dst_idx = self.board.index_of.get((src_cell.q + dq, src_cell.r + dr))
        if dst_idx is None:
            return False
        return not self.board.cells[dst_idx].occupied

    def _has_continuing_jump(self, pin_id):
        pin = self.pins[self.current_player][pin_id]
        for direction_idx in range(len(DIRECTIONS)):
            if self._is_legal_submove(pin, direction_idx, is_jump=True):
                return True
        return False

    def _has_player_won(self, player):
        target_colour = self.board.colour_opposites[COLOURS[player]]
        target_cells = set(self.board.axial_of_colour(target_colour))
        current_positions = {p.axialindex for p in self.pins[player]}
        return current_positions == target_cells

    def _switch_turn(self):
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()
        self.current_player = 1 - self.current_player


def _goal_centroid(match, player):
    target_colour = match.board.colour_opposites[COLOURS[player]]
    target_indices = match.board.axial_of_colour(target_colour)
    target_cells = [match.board.cells[i] for i in target_indices]
    goal_q = sum(c.q for c in target_cells) / len(target_cells)
    goal_r = sum(c.r for c in target_cells) / len(target_cells)
    return goal_q, goal_r, set(target_indices)


def _distance_to_goal(match, idx, goal_q, goal_r):
    cell = match.board.cells[idx]
    return (
        abs(cell.q - goal_q)
        + abs(cell.q + cell.r - goal_q - goal_r)
        + abs(cell.r - goal_r)
    ) / 2


def _hex_distance(match, src_idx, dst_idx):
    src = match.board.cells[src_idx]
    dst = match.board.cells[dst_idx]
    return (
        abs(src.q - dst.q)
        + abs((src.q + src.r) - (dst.q + dst.r))
        + abs(src.r - dst.r)
    ) / 2


def _assignment_cost(match, positions, target_indices):
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
            dist = _hex_distance(match, positions[i], target_idx)
            best = min(best, dist + dp(i + 1, used_mask | (1 << target_i)))
        return best

    return dp(0, 0)


def greedy_action(match):
    mask = match.action_masks()
    legal_actions = np.flatnonzero(mask)
    if len(legal_actions) == 0:
        return END_TURN_ACTION

    player = match.current_player
    goal_q, goal_r, target_index_set = _goal_centroid(match, player)
    current_positions = [pin.axialindex for pin in match.pins[player]]
    current_assignment_cost = _assignment_cost(match, current_positions, target_index_set)

    best_score = -float("inf")
    best_actions = []

    for action in legal_actions:
        action = int(action)
        if action == END_TURN_ACTION:
            score = -0.25
        else:
            pin_id, direction_idx, is_jump = match._decode_action(action)
            pin = match.pins[player][pin_id]
            src_idx = pin.axialindex
            dest_idx = match._destination_for_submove(src_idx, direction_idx, is_jump)
            if dest_idx is None:
                continue

            src_dist = _distance_to_goal(match, src_idx, goal_q, goal_r)
            dest_dist = _distance_to_goal(match, dest_idx, goal_q, goal_r)
            progress = src_dist - dest_dist

            next_positions = list(current_positions)
            next_positions[pin_id] = dest_idx
            assignment_gain = current_assignment_cost - _assignment_cost(
                match, next_positions, target_index_set
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
        return END_TURN_ACTION
    return int(np.random.choice(best_actions))


def mcts_action(match, simulations):
    state = SearchState(
        board=match.board,
        pins=match.pins,
        current_player=match.current_player,
        step_count=match.step_count,
        max_steps=match.max_steps,
        winner=match.winner,
        active_jump_pin_id=match.active_jump_pin_id,
        jump_visited_cells=match.jump_visited_cells,
    )
    searcher = NeuralGuidedMCTS(guide=NeuralGuide(model=None), simulations=simulations, cpuct=1.5)
    action, _ = searcher.run(state)
    if action is None:
        return END_TURN_ACTION
    if action.is_end_turn:
        return END_TURN_ACTION
    return match._encode_action(action.pin_id, action.direction_idx, action.is_jump)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a PPO model directly on the teacher board code."
    )
    parser.add_argument("--model-path", required=True, help="Path to PPO zip model.")
    parser.add_argument(
        "--opponent",
        choices=["greedy", "mcts", "ppo"],
        default="greedy",
        help="Opponent type for player 1.",
    )
    parser.add_argument(
        "--opponent-model",
        help="Optional second PPO zip when --opponent ppo is used.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--mcts-simulations", type=int, default=60)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--ppo-player",
        type=int,
        choices=[0, 1],
        default=0,
        help="Seat for the main PPO model: 0 is red, 1 is blue.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = MaskablePPO.load(args.model_path)
    opponent_model = MaskablePPO.load(args.opponent_model) if args.opponent_model else None
    if args.opponent == "ppo" and opponent_model is None:
        raise ValueError("--opponent-model is required when --opponent ppo is used.")
    match = TeacherBoardMatch(max_steps=args.max_steps)
    obs, info = match.observe()
    seat_labels = ["", ""]
    seat_labels[args.ppo_player] = "PPO"
    opponent_player = 1 - args.ppo_player
    seat_labels[opponent_player] = args.opponent.upper() if args.opponent == "mcts" else args.opponent

    print(f"Loaded PPO model: {Path(args.model_path).name}")
    print(f"PPO player: {args.ppo_player} ({COLOURS[args.ppo_player]})")
    if args.opponent == "ppo":
        print(f"Opponent PPO model: {Path(args.opponent_model).name}")
    elif args.opponent == "mcts":
        print(f"Opponent: MCTS baseline ({args.mcts_simulations} simulations)")
    else:
        print(f"Opponent: greedy baseline as player {opponent_player} ({COLOURS[opponent_player]})")
    print(f"Seats: red = {seat_labels[0]}, blue = {seat_labels[1]}")

    done = False
    truncated = False
    while not done:
        if args.render:
            os.system("clear")
            print(
                f"\nStep {match.step_count} | player {match.current_player} "
                f"({COLOURS[match.current_player]} = {seat_labels[match.current_player]})"
            )
            match.render()

        if match.current_player == args.ppo_player:
            action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
        else:
            if args.opponent == "ppo":
                action, _ = opponent_model.predict(
                    obs, action_masks=info["action_mask"], deterministic=True
                )
            elif args.opponent == "mcts":
                action = mcts_action(match, simulations=args.mcts_simulations)
            else:
                action = greedy_action(match)

        obs, info, terminated, truncated, extra = match.step(action)
        done = terminated or truncated
        if args.delay > 0:
            time.sleep(args.delay)

    if args.render:
        os.system("clear")
    print(f"\nFinished after {match.step_count} steps")
    print(f"Seats: red = {seat_labels[0]}, blue = {seat_labels[1]}")
    match.render()
    if "winner" in extra:
        winner = extra["winner"]
        print(f"Winner: player {winner} ({COLOURS[winner]} = {seat_labels[winner]})")
    elif truncated:
        print("Game ended by max steps.")


if __name__ == "__main__":
    main()
