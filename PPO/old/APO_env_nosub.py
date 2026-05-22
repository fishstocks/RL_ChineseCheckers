import numpy as np
import io
import os
import sys
from contextlib import redirect_stdout
import gymnasium as gym
from gymnasium import spaces

TEACHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "single system"))
if TEACHER_DIR not in sys.path:
    sys.path.append(TEACHER_DIR)

from checkers_board import HexBoard
from checkers_pins import Pin


# ----------------- Environment Wrapper ----------------- #
# This class turns the teacher's Chinese Checkers code into a Gym environment
# that PPO can train on.
class ChineseCheckersEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}
    N_PINS = 10

    def __init__(self, render_mode=None, max_steps=300):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # ----------------- Board / Size Setup ----------------- #
        # Build a temporary board once so we can find out how many cells exist.
        with redirect_stdout(io.StringIO()):
            temp_board = HexBoard(R=4, hole_radius=16, spacing=34)
        self.N_CELLS = len(temp_board.cells)

        # ----------------- Action Space ----------------- #
        # An action means:
        # 1. choose which pin to move
        # 2. choose which cell to move it to
        # We also add one extra "pass" action in case a player has no legal move.
        self.PASS_ACTION = self.N_PINS * self.N_CELLS
        self.ACTION_DIM = self.PASS_ACTION + 1
        self.action_space = spaces.Discrete(self.ACTION_DIM)

        # ----------------- Observation Space ----------------- #
        # The observation is a flat vector with 3 layers:
        # - my pins
        # - enemy pins
        # - my target triangle
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.N_CELLS * 3,), dtype=np.float32
        )

        # ----------------- Game State ----------------- #
        # We only train the 1v1 version here: red vs blue.
        self.colours = ["red", "blue"]
        
        # These hold the current live game state.
        self.board = None
        self.pins = [[], []]
        self.current_player = 0
        self.done = True
        self.step_count = 0
        self.state_visit_counts = {}

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

        with redirect_stdout(io.StringIO()):
            for player, colour in enumerate(self.colours):
                axials = self.board.axial_of_colour(colour)
                self.pins[player] = [
                    Pin(self.board, axials[i], id=i, color=colour)
                    for i in range(self.N_PINS)
                ]
            
        self._record_current_state()
        return self._get_obs(), {"action_mask": self.action_masks()}

    # ----------------- Step / One Move ----------------- #
    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        legal_mask = self.action_masks()

        # ----------------- Pass Action ----------------- #
        if action == self.PASS_ACTION:
            if legal_mask[:self.PASS_ACTION].any():
                return self._get_obs(), -0.5, False, False, {
                    "illegal": True,
                    "action_mask": legal_mask
                }

            self.step_count += 1

            if self.step_count >= self.max_steps:
                self.done = True
                return self._get_obs(), -0.1, False, True, {
                    "reason": "max_steps",
                    "action_mask": self.action_masks()
                }

            self.current_player = 1 - self.current_player
            return self._get_obs(), -0.1, False, False, {"action_mask": self.action_masks()}

        # Decode the flat action id into:
        # - which pin to move
        # - which destination cell to use
        pin_id = action // self.N_CELLS
        dest_idx = action % self.N_CELLS

        # ----------------- Action Validation ----------------- #
        if pin_id < 0 or pin_id >= self.N_PINS:
            return self._get_obs(), -0.5, False, False, {"invalid": True, "action_mask": legal_mask}

        pin = self.pins[self.current_player][pin_id]
        legal_dests = pin.getPossibleMoves()

        if dest_idx not in legal_dests:
            return self._get_obs(), -0.5, False, False, {"illegal": True, "action_mask": legal_mask}

        self.step_count += 1

        # ----------------- Goal / Distance Info ----------------- #
        target_colour = self.board.colour_opposites[self.colours[self.current_player]]
        target_indices = self.board.axial_of_colour(target_colour)
        target_index_set = set(target_indices)
        target_cells_objs = [self.board.cells[i] for i in target_indices]
        
        goal_q = sum(c.q for c in target_cells_objs) / len(target_cells_objs)
        goal_r = sum(c.r for c in target_cells_objs) / len(target_cells_objs)

        def get_dist_to_centroid(idx):
            c = self.board.cells[idx]
            return (abs(c.q - goal_q) + abs(c.q + c.r - goal_q - goal_r) + abs(c.r - goal_r)) / 2

        dist_before = get_dist_to_centroid(pin.axialindex)
        src_idx = pin.axialindex
        src_cell = self.board.cells[src_idx]
        dst_cell = self.board.cells[dest_idx]

        with redirect_stdout(io.StringIO()):
            pin.placePin(dest_idx)
        dist_after = get_dist_to_centroid(dest_idx)

        # ----------------- Reward Shaping ----------------- #
        step_penalty = -0.01
        progress_reward = (dist_before - dist_after) * 0.1
        reward = step_penalty + progress_reward

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

        visits = self._record_current_state()
        if visits > 1:
            reward -= 0.05 * (visits - 1)

        # ----------------- Win Check ----------------- #
        if self._has_player_won(self.current_player):
            self.done = True
            return self._get_obs(), 10.0, True, False, {
                "winner": self.current_player, 
                "action_mask": self.action_masks()
            }

        # ----------------- Max Step Limit ----------------- #
        if self.step_count >= self.max_steps:
            self.done = True
            return self._get_obs(), 0.0, False, True, {
                "reason": "max_steps", 
                "action_mask": self.action_masks()
            }

        # ----------------- Turn Switch ----------------- #
        self.current_player = 1 - self.current_player
        return self._get_obs(), reward, False, False, {"action_mask": self.action_masks()}

    # ----------------- Observation Builder ----------------- #
    def _get_obs(self) -> np.ndarray:
        l1 = np.zeros(self.N_CELLS, dtype=np.float32)
        for p in self.pins[self.current_player]:
            l1[p.axialindex] = 1
            
        l2 = np.zeros(self.N_CELLS, dtype=np.float32)
        for p in self.pins[1 - self.current_player]:
            l2[p.axialindex] = 1
            
        l3 = np.zeros(self.N_CELLS, dtype=np.float32)
        target_colour = self.board.colour_opposites[self.colours[self.current_player]]
        for idx in self.board.axial_of_colour(target_colour):
            l3[idx] = 1
            
        return np.concatenate([l1, l2, l3]).astype(np.float32)

    # ----------------- Legal Action Mask ----------------- #
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.ACTION_DIM, dtype=bool)
        
        if self.board is None:
            mask[:] = True
            return mask
        
        for pin in self.pins[self.current_player]:
            for dest in pin.getPossibleMoves():
                mask[pin.id * self.N_CELLS + dest] = True

        if not mask.any():
            mask[self.PASS_ACTION] = True

        return mask

    # ----------------- Loop Tracking ----------------- #
    def _state_key(self):
        red_positions = tuple(sorted(p.axialindex for p in self.pins[0]))
        blue_positions = tuple(sorted(p.axialindex for p in self.pins[1]))
        return (self.current_player, red_positions, blue_positions)

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
