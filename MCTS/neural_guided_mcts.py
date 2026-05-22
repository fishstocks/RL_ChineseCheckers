import argparse
import io
import math
import os
import random
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np
import torch
from sb3_contrib import MaskablePPO


TEACHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "single system"))
if TEACHER_DIR not in sys.path:
    sys.path.append(TEACHER_DIR)

from checkers_board import HexBoard
from checkers_pins import Pin


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


@dataclass(frozen=True)
class SubmoveAction:
    pin_id: int
    direction_idx: int
    is_jump: bool
    is_end_turn: bool = False

    def to_text(self):
        if self.is_end_turn:
            return "END_TURN"
        move_type = "jump" if self.is_jump else "step"
        return f"pin {self.pin_id}, dir {self.direction_idx}, {move_type}"


class SearchState:
    def __init__(
        self,
        board,
        pins,
        current_player=0,
        step_count=0,
        max_steps=300,
        winner=None,
        active_jump_pin_id=None,
        jump_visited_cells=None,
    ):
        self.board = board
        self.pins = pins
        self.current_player = current_player
        self.step_count = step_count
        self.max_steps = max_steps
        self.winner = winner
        self.active_jump_pin_id = active_jump_pin_id
        self.jump_visited_cells = set(jump_visited_cells or [])

        self.n_cells = len(self.board.cells)
        self.actions_per_pin = len(DIRECTIONS) * 2
        self.end_turn_action = N_PINS * self.actions_per_pin
        self.action_dim = self.end_turn_action + 1

    @classmethod
    def new_game(cls, max_steps=300):
        with redirect_stdout(io.StringIO()):
            board = HexBoard(R=4, hole_radius=16, spacing=34)
        pins = [[], []]
        with redirect_stdout(io.StringIO()):
            for player, colour in enumerate(COLOURS):
                axials = board.axial_of_colour(colour)
                pins[player] = [Pin(board, axials[i], id=i, color=colour) for i in range(N_PINS)]
        return cls(board=board, pins=pins, max_steps=max_steps)

    def clone(self):
        with redirect_stdout(io.StringIO()):
            new_board = HexBoard(R=4, hole_radius=16, spacing=34)
        new_pins = [[], []]
        with redirect_stdout(io.StringIO()):
            for player in range(2):
                for pin in self.pins[player]:
                    new_pins[player].append(Pin(new_board, pin.axialindex, id=pin.id, color=pin.color))
        return SearchState(
            board=new_board,
            pins=new_pins,
            current_player=self.current_player,
            step_count=self.step_count,
            max_steps=self.max_steps,
            winner=self.winner,
            active_jump_pin_id=self.active_jump_pin_id,
            jump_visited_cells=self.jump_visited_cells,
        )

    def is_terminal(self):
        return self.winner is not None or self.step_count >= self.max_steps

    def _encode_action(self, pin_id, direction_idx, is_jump):
        return pin_id * self.actions_per_pin + direction_idx * 2 + int(is_jump)

    def _decode_action(self, action_idx):
        if action_idx == self.end_turn_action:
            return SubmoveAction(0, 0, False, is_end_turn=True)
        pin_id = action_idx // self.actions_per_pin
        local = action_idx % self.actions_per_pin
        direction_idx = local // 2
        is_jump = bool(local % 2)
        return SubmoveAction(pin_id, direction_idx, is_jump)

    def _destination_for_submove(self, src_idx, direction_idx, is_jump):
        src_cell = self.board.cells[src_idx]
        dq, dr = DIRECTIONS[direction_idx]
        mult = 2 if is_jump else 1
        return self.board.index_of.get((src_cell.q + dq * mult, src_cell.r + dr * mult))

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

    def has_player_won(self, player):
        target_colour = self.board.colour_opposites[COLOURS[player]]
        target_cells = set(self.board.axial_of_colour(target_colour))
        current_positions = {p.axialindex for p in self.pins[player]}
        return current_positions == target_cells

    def _has_continuing_jump(self, pin_id):
        pin = self.pins[self.current_player][pin_id]
        for direction_idx in range(len(DIRECTIONS)):
            if self._is_legal_submove(pin, direction_idx, is_jump=True):
                return True
        return False

    def _switch_turn(self):
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()
        self.current_player = 1 - self.current_player

    def legal_action_mask(self):
        mask = np.zeros(self.action_dim, dtype=bool)
        if self.is_terminal():
            return mask

        if self.active_jump_pin_id is not None:
            pin = self.pins[self.current_player][self.active_jump_pin_id]
            for direction_idx in range(len(DIRECTIONS)):
                if self._is_legal_submove(pin, direction_idx, is_jump=True):
                    mask[self._encode_action(pin.id, direction_idx, True)] = True
            mask[self.end_turn_action] = True
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
            mask[self.end_turn_action] = True
        return mask

    def legal_actions(self):
        mask = self.legal_action_mask()
        return [self._decode_action(i) for i in np.flatnonzero(mask)]

    def apply_action(self, action):
        next_state = self.clone()
        next_state.step_count += 1

        if action.is_end_turn:
            if next_state.step_count < next_state.max_steps:
                next_state._switch_turn()
            return next_state

        pin = next_state.pins[next_state.current_player][action.pin_id]
        src_idx = pin.axialindex
        dest_idx = next_state._destination_for_submove(src_idx, action.direction_idx, action.is_jump)
        with redirect_stdout(io.StringIO()):
            pin.placePin(dest_idx)

        if action.is_jump:
            if next_state.active_jump_pin_id is None:
                next_state.active_jump_pin_id = action.pin_id
                next_state.jump_visited_cells = {src_idx, dest_idx}
            else:
                next_state.jump_visited_cells.add(dest_idx)

        if next_state.has_player_won(next_state.current_player):
            next_state.winner = next_state.current_player
            return next_state

        if next_state.step_count >= next_state.max_steps:
            return next_state

        if not action.is_jump:
            next_state._switch_turn()
            return next_state

        if not next_state._has_continuing_jump(action.pin_id):
            next_state._switch_turn()

        return next_state

    def observe(self):
        l1 = np.zeros(self.n_cells, dtype=np.float32)
        for p in self.pins[self.current_player]:
            l1[p.axialindex] = 1.0

        l2 = np.zeros(self.n_cells, dtype=np.float32)
        for p in self.pins[1 - self.current_player]:
            l2[p.axialindex] = 1.0

        l3 = np.zeros(self.n_cells, dtype=np.float32)
        target_colour = self.board.colour_opposites[COLOURS[self.current_player]]
        for idx in self.board.axial_of_colour(target_colour):
            l3[idx] = 1.0

        l4 = np.zeros(self.n_cells, dtype=np.float32)
        if self.active_jump_pin_id is not None:
            active_pin = self.pins[self.current_player][self.active_jump_pin_id]
            l4[active_pin.axialindex] = 1.0

        return np.concatenate([l1, l2, l3, l4]).astype(np.float32)

    def total_goal_distance(self, player):
        target_colour = self.board.colour_opposites[COLOURS[player]]
        target_indices = self.board.axial_of_colour(target_colour)
        target_cells = [self.board.cells[i] for i in target_indices]
        goal_q = sum(c.q for c in target_cells) / len(target_cells)
        goal_r = sum(c.r for c in target_cells) / len(target_cells)

        total = 0.0
        for pin in self.pins[player]:
            cell = self.board.cells[pin.axialindex]
            total += (
                abs(cell.q - goal_q)
                + abs(cell.q + cell.r - goal_q - goal_r)
                + abs(cell.r - goal_r)
            ) / 2
        return total

    def heuristic_value(self):
        if self.winner is not None:
            return 1.0
        if self.step_count >= self.max_steps:
            return 0.0

        me = self.current_player
        opp = 1 - me
        my_dist = self.total_goal_distance(me)
        opp_dist = self.total_goal_distance(opp)
        distance_score = (opp_dist - my_dist) / 100.0
        return max(-1.0, min(1.0, distance_score))

    def terminal_value(self):
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == self.current_player else -1.0

    def render(self):
        all_pins = self.pins[0] + self.pins[1]
        self.board.print_ascii(pins=all_pins)


class NeuralGuide:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def from_path(cls, model_path=None):
        if model_path is None:
            return cls(model=None)
        model = MaskablePPO.load(model_path)
        return cls(model=model)

    def evaluate(self, state: SearchState):
        legal_mask = state.legal_action_mask()
        if not legal_mask.any():
            return np.zeros(state.action_dim, dtype=np.float32), state.terminal_value()

        if self.model is None:
            priors = legal_mask.astype(np.float32)
            priors /= priors.sum()
            return priors, state.heuristic_value()

        obs = state.observe()
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            raw_probs = dist.distribution.probs.squeeze(0).detach().cpu().numpy()
            value = self.model.policy.predict_values(obs_tensor).squeeze().item()

        priors = np.zeros_like(raw_probs, dtype=np.float32)
        priors[legal_mask] = raw_probs[legal_mask]
        total = priors.sum()
        if total <= 0:
            priors = legal_mask.astype(np.float32)
            priors /= priors.sum()
        else:
            priors /= total

        value = float(np.clip(value, -1.0, 1.0))
        return priors.astype(np.float32), value


class TreeNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}
        self.unexpanded_actions = None
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q_value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_expanded(self):
        return self.unexpanded_actions is not None


class NeuralGuidedMCTS:
    def __init__(self, guide, simulations=100, cpuct=1.5):
        self.guide = guide
        self.simulations = simulations
        self.cpuct = cpuct

    def _expand(self, node, priors):
        legal_actions = node.state.legal_actions()
        node.unexpanded_actions = []
        for action in legal_actions:
            if action.is_end_turn:
                action_idx = node.state.end_turn_action
            else:
                action_idx = node.state._encode_action(action.pin_id, action.direction_idx, action.is_jump)
            prior = float(priors[action_idx])
            node.unexpanded_actions.append((action, prior))

    def _select_child(self, node):
        best_score = -float("inf")
        best_pair = None
        parent_visits = max(1, node.visits)

        for action, child in node.children.items():
            u = self.cpuct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
            score = child.q_value + u
            if score > best_score:
                best_score = score
                best_pair = (action, child)
        return best_pair

    def run(self, root_state):
        root = TreeNode(root_state.clone())
        root_priors, _ = self.guide.evaluate(root.state)
        self._expand(root, root_priors)

        if not root.unexpanded_actions:
            return None, root

        for _ in range(self.simulations):
            node = root

            while node.is_expanded() and len(node.unexpanded_actions) == 0 and node.children and not node.state.is_terminal():
                _, node = self._select_child(node)

            if node.state.is_terminal():
                value = node.state.terminal_value()
            else:
                if node.unexpanded_actions is None:
                    priors, value = self.guide.evaluate(node.state)
                    self._expand(node, priors)
                else:
                    action, prior = node.unexpanded_actions.pop(random.randrange(len(node.unexpanded_actions)))
                    child_state = node.state.apply_action(action)
                    child = TreeNode(child_state, parent=node, action=action, prior=prior)
                    node.children[action] = child
                    node = child

                if node.state.is_terminal():
                    value = node.state.terminal_value()
                else:
                    priors, value = self.guide.evaluate(node.state)
                    self._expand(node, priors)

            while node is not None:
                node.visits += 1
                node.value_sum += value
                value = -value
                node = node.parent

        best_action, _ = max(root.children.items(), key=lambda item: item[1].visits)
        return best_action, root


def print_root_stats(root, top_k=5):
    ranked = sorted(root.children.items(), key=lambda item: item[1].visits, reverse=True)
    print("Top searched actions:")
    for action, child in ranked[:top_k]:
        print(
            f"  {action.to_text()} | prior={child.prior:.3f} "
            f"| visits={child.visits} | q={child.q_value:.3f}"
        )


def play_game(model_path=None, simulations=120, delay=0.35, max_steps=300):
    state = SearchState.new_game(max_steps=max_steps)
    guide = NeuralGuide.from_path(model_path)
    searcher = NeuralGuidedMCTS(guide=guide, simulations=simulations, cpuct=1.5)

    while not state.is_terminal():
        os.system("clear")
        print(f"Step {state.step_count} | current player: {state.current_player}")
        state.render()
        action, root = searcher.run(state)
        if action is None:
            print("No legal actions available.")
            break
        print()
        print_root_stats(root)
        print(f"\nChosen action: {action.to_text()}")
        time.sleep(delay)
        state = state.apply_action(action)

    os.system("clear")
    print(f"Game finished after {state.step_count} steps")
    state.render()
    if state.winner is not None:
        print(f"Winner: player {state.winner}")
    else:
        print("No winner recorded.")


def main():
    parser = argparse.ArgumentParser(description="Neural-guided MCTS for 1v1 Chinese Checkers.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional PPO .zip model path.")
    parser.add_argument("--simulations", type=int, default=120, help="MCTS simulations per move.")
    parser.add_argument("--delay", type=float, default=0.35, help="Delay between moves.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum game length.")
    args = parser.parse_args()

    play_game(
        model_path=args.model_path,
        simulations=args.simulations,
        delay=args.delay,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
