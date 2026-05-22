import io
import os
import sys
from contextlib import redirect_stdout

import gymnasium as gym
import numpy as np
from gymnasium import spaces


TEACHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "single system"))
if TEACHER_DIR not in sys.path:
    sys.path.append(TEACHER_DIR)

from checkers_board import HexBoard
from checkers_pins import Pin


# ----------------- Environment Wrapper ----------------- #
# This PPO environment uses submoves instead of "pick any final destination".
# One action means:
# - choose a pin
# - choose a direction
# - choose step or jump
# or choose a special end-turn action.
class ChineseCheckersEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    N_PINS = 10
    DIRECTIONS = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
    ]

    def __init__(self, render_mode=None, max_steps=300):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # ----------------- Board / Size Setup ----------------- #
        with redirect_stdout(io.StringIO()):
            temp_board = HexBoard(R=4, hole_radius=16, spacing=34)
        self.N_CELLS = len(temp_board.cells)

        # ----------------- Action Space ----------------- #
        # Per pin:
        # - 6 step directions
        # - 6 jump directions
        # Plus one global END_TURN action.
        self.ACTIONS_PER_PIN = len(self.DIRECTIONS) * 2
        self.END_TURN_ACTION = self.N_PINS * self.ACTIONS_PER_PIN
        self.ACTION_DIM = self.END_TURN_ACTION + 1
        self.action_space = spaces.Discrete(self.ACTION_DIM)

        # ----------------- Observation Space ----------------- #
        # Observation layers:
        # - my pins
        # - enemy pins
        # - my target triangle
        # - active jump pin (if this turn is in the middle of a jump chain)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.N_CELLS * 4,),
            dtype=np.float32,
        )

        self.colours = ["red", "blue"]

        # ----------------- Game State ----------------- #
        self.board = None
        self.pins = [[], []]
        self.current_player = 0
        self.done = True
        self.step_count = 0
        self.state_visit_counts = {}

        # Submove-specific state.
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()

    # ----------------- Reset / New Game ----------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        with redirect_stdout(io.StringIO()):
            self.board = HexBoard(R=4, hole_radius=16, spacing=34)

        self.pins = [[], []]
        self.current_player = 0
        self.done = False
        self.step_count = 0
        self.state_visit_counts = {}
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()

        with redirect_stdout(io.StringIO()):
            for player, colour in enumerate(self.colours):
                axials = self.board.axial_of_colour(colour)
                self.pins[player] = [
                    Pin(self.board, axials[i], id=i, color=colour)
                    for i in range(self.N_PINS)
                ]

        self._record_current_state()
        return self._get_obs(), {"action_mask": self.action_masks()}

    # ----------------- Step / One Submove ----------------- #
    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        legal_mask = self.action_masks()
        if action < 0 or action >= self.ACTION_DIM or not legal_mask[action]:
            return self._get_obs(), -0.5, False, False, {
                "illegal": True,
                "action_mask": legal_mask,
            }

        # ----------------- End Turn ----------------- #
        if action == self.END_TURN_ACTION:
            self.step_count += 1
            reward = -0.02

            visits = self._record_current_state()
            if visits > 1:
                reward -= 0.05 * (visits - 1)

            if self.step_count >= self.max_steps:
                self.done = True
                return self._get_obs(), reward, False, True, {
                    "reason": "max_steps",
                    "action_mask": self.action_masks(),
                }

            self._switch_turn()
            return self._get_obs(), reward, False, False, {"action_mask": self.action_masks()}

        pin_id, direction_idx, is_jump = self._decode_action(action)
        pin = self.pins[self.current_player][pin_id]
        src_idx = pin.axialindex
        src_cell = self.board.cells[src_idx]
        dest_idx = self._destination_for_submove(src_idx, direction_idx, is_jump)
        dst_cell = self.board.cells[dest_idx]

        self.step_count += 1

        # ----------------- Goal / Distance Info ----------------- #
        target_colour = self.board.colour_opposites[self.colours[self.current_player]]
        target_indices = self.board.axial_of_colour(target_colour)
        target_index_set = set(target_indices)
        target_cells = [self.board.cells[i] for i in target_indices]

        goal_q = sum(c.q for c in target_cells) / len(target_cells)
        goal_r = sum(c.r for c in target_cells) / len(target_cells)

        def get_dist_to_centroid(idx):
            c = self.board.cells[idx]
            return (abs(c.q - goal_q) + abs(c.q + c.r - goal_q - goal_r) + abs(c.r - goal_r)) / 2

        dist_before = get_dist_to_centroid(src_idx)
        with redirect_stdout(io.StringIO()):
            pin.placePin(dest_idx)
        dist_after = get_dist_to_centroid(dest_idx)

        # ----------------- Reward Shaping ----------------- #
        reward = -0.01
        reward += (dist_before - dist_after) * 0.1

        if src_idx not in target_index_set and dest_idx in target_index_set:
            reward += 0.25
        elif src_idx in target_index_set and dest_idx not in target_index_set:
            reward -= 0.25

        if self.colours[self.current_player] == "red":
            if dst_cell.r < src_cell.r:
                reward += 0.02
            elif dst_cell.r > src_cell.r:
                reward -= 0.02
        else:
            if dst_cell.r > src_cell.r:
                reward += 0.02
            elif dst_cell.r < src_cell.r:
                reward -= 0.02

        # ----------------- Jump Chain Handling ----------------- #
        if is_jump:
            if self.active_jump_pin_id is None:
                self.active_jump_pin_id = pin_id
                self.jump_visited_cells = {src_idx, dest_idx}
            else:
                self.jump_visited_cells.add(dest_idx)

        visits = self._record_current_state()
        if visits > 1:
            reward -= 0.05 * (visits - 1)

        # ----------------- Win Check ----------------- #
        if self._has_player_won(self.current_player):
            self.done = True
            return self._get_obs(), 10.0, True, False, {
                "winner": self.current_player,
                "action_mask": self.action_masks(),
            }

        if self.step_count >= self.max_steps:
            self.done = True
            return self._get_obs(), reward, False, True, {
                "reason": "max_steps",
                "action_mask": self.action_masks(),
            }

        # Step submoves always end the turn immediately.
        if not is_jump:
            self._switch_turn()
            return self._get_obs(), reward, False, False, {"action_mask": self.action_masks()}

        # If a jump chain has no continuation, end the turn automatically.
        if not self._has_continuing_jump(pin_id):
            self._switch_turn()

        return self._get_obs(), reward, False, False, {"action_mask": self.action_masks()}

    # ----------------- Observation Builder ----------------- #
    def _get_obs(self) -> np.ndarray:
        l1 = np.zeros(self.N_CELLS, dtype=np.float32)
        for p in self.pins[self.current_player]:
            l1[p.axialindex] = 1.0

        l2 = np.zeros(self.N_CELLS, dtype=np.float32)
        for p in self.pins[1 - self.current_player]:
            l2[p.axialindex] = 1.0

        l3 = np.zeros(self.N_CELLS, dtype=np.float32)
        target_colour = self.board.colour_opposites[self.colours[self.current_player]]
        for idx in self.board.axial_of_colour(target_colour):
            l3[idx] = 1.0

        l4 = np.zeros(self.N_CELLS, dtype=np.float32)
        if self.active_jump_pin_id is not None:
            active_pin = self.pins[self.current_player][self.active_jump_pin_id]
            l4[active_pin.axialindex] = 1.0

        return np.concatenate([l1, l2, l3, l4]).astype(np.float32)

    # ----------------- Legal Action Mask ----------------- #
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.ACTION_DIM, dtype=bool)

        if self.board is None:
            mask[:] = True
            return mask

        # During a jump chain, only the same pin can keep jumping or end the turn.
        if self.active_jump_pin_id is not None:
            pin = self.pins[self.current_player][self.active_jump_pin_id]
            jump_found = False
            for direction_idx in range(len(self.DIRECTIONS)):
                if self._is_legal_submove(pin, direction_idx, is_jump=True):
                    mask[self._encode_action(pin.id, direction_idx, True)] = True
                    jump_found = True
            mask[self.END_TURN_ACTION] = True
            if not jump_found:
                mask[self.END_TURN_ACTION] = True
            return mask

        has_any_move = False
        for pin in self.pins[self.current_player]:
            for direction_idx in range(len(self.DIRECTIONS)):
                if self._is_legal_submove(pin, direction_idx, is_jump=False):
                    mask[self._encode_action(pin.id, direction_idx, False)] = True
                    has_any_move = True
                if self._is_legal_submove(pin, direction_idx, is_jump=True):
                    mask[self._encode_action(pin.id, direction_idx, True)] = True
                    has_any_move = True

        if not has_any_move:
            mask[self.END_TURN_ACTION] = True

        return mask

    # ----------------- Submove Helpers ----------------- #
    def _encode_action(self, pin_id, direction_idx, is_jump):
        return pin_id * self.ACTIONS_PER_PIN + direction_idx * 2 + int(is_jump)

    def _decode_action(self, action):
        pin_id = action // self.ACTIONS_PER_PIN
        local = action % self.ACTIONS_PER_PIN
        direction_idx = local // 2
        is_jump = bool(local % 2)
        return pin_id, direction_idx, is_jump

    def _destination_for_submove(self, src_idx, direction_idx, is_jump):
        src_cell = self.board.cells[src_idx]
        dq, dr = self.DIRECTIONS[direction_idx]
        multiplier = 2 if is_jump else 1
        q = src_cell.q + dq * multiplier
        r = src_cell.r + dr * multiplier
        return self.board.index_of.get((q, r))

    def _is_legal_submove(self, pin, direction_idx, is_jump):
        if self.active_jump_pin_id is not None and pin.id != self.active_jump_pin_id:
            return False

        src_idx = pin.axialindex
        src_cell = self.board.cells[src_idx]
        dq, dr = self.DIRECTIONS[direction_idx]

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

        # Steps are not allowed in the middle of a jump chain.
        if self.active_jump_pin_id is not None:
            return False

        dst_idx = self.board.index_of.get((src_cell.q + dq, src_cell.r + dr))
        if dst_idx is None:
            return False
        return not self.board.cells[dst_idx].occupied

    def _has_continuing_jump(self, pin_id):
        pin = self.pins[self.current_player][pin_id]
        for direction_idx in range(len(self.DIRECTIONS)):
            if self._is_legal_submove(pin, direction_idx, is_jump=True):
                return True
        return False

    def _switch_turn(self):
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()
        self.current_player = 1 - self.current_player

    # ----------------- Loop Tracking ----------------- #
    def _state_key(self):
        red_positions = tuple(sorted(p.axialindex for p in self.pins[0]))
        blue_positions = tuple(sorted(p.axialindex for p in self.pins[1]))
        jump_state = (
            self.active_jump_pin_id,
            tuple(sorted(self.jump_visited_cells)),
        )
        return (self.current_player, red_positions, blue_positions, jump_state)

    def _record_current_state(self) -> int:
        key = self._state_key()
        self.state_visit_counts[key] = self.state_visit_counts.get(key, 0) + 1
        return self.state_visit_counts[key]

    # ----------------- Win Condition ----------------- #
    def _has_player_won(self, player: int) -> bool:
        target_colour = self.board.colour_opposites[self.colours[player]]
        target_cells = set(self.board.axial_of_colour(target_colour))
        current_positions = {p.axialindex for p in self.pins[player]}
        return current_positions == target_cells

    # ----------------- Rendering ----------------- #
    def render(self):
        if self.render_mode == "ansi" and self.board:
            self.board.print_ascii(pins=self.pins[0] + self.pins[1])
