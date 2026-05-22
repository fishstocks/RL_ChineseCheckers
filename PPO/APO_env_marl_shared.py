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


class ChineseCheckersEnvMARL(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    PLAYER_COLOUR_SETS = {
        2: ["red", "blue"],
        3: ["red", "yellow", "lawn green"],
        4: ["red", "blue", "yellow", "purple"],
        6: ["red", "blue", "yellow", "lawn green", "purple", "gray0"],
    }
    DIRECTIONS = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
    ]
    N_PINS = 10

    def __init__(self, num_players=4, render_mode=None, max_steps=400):
        super().__init__()
        if num_players not in (2, 3, 4, 6):
            raise ValueError("num_players must be one of: 2, 3, 4, 6")

        self.num_players = num_players
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.player_colours = list(self.PLAYER_COLOUR_SETS[num_players])

        with redirect_stdout(io.StringIO()):
            temp_board = HexBoard(R=4, hole_radius=16, spacing=34)
        self.N_CELLS = len(temp_board.cells)

        # pin x (6 step directions + 6 jump directions) + end-turn
        self.ACTIONS_PER_PIN = len(self.DIRECTIONS) * 2
        self.END_TURN_ACTION = self.N_PINS * self.ACTIONS_PER_PIN
        self.ACTION_DIM = self.END_TURN_ACTION + 1
        self.action_space = spaces.Discrete(self.ACTION_DIM)

        # Observation layers:
        # - current player's pins
        # - all opponent pins merged together
        # - current player's target triangle
        # - active jump pin during a jump chain
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.N_CELLS * 4,),
            dtype=np.float32,
        )

        self.board = None
        self.pins = []
        self.current_player = 0
        self.done = True
        self.step_count = 0
        self.state_visit_counts = {}
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        with redirect_stdout(io.StringIO()):
            self.board = HexBoard(R=4, hole_radius=16, spacing=34)

        self.pins = [[] for _ in range(self.num_players)]
        self.current_player = 0
        self.done = False
        self.step_count = 0
        self.state_visit_counts = {}
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()

        with redirect_stdout(io.StringIO()):
            for player, colour in enumerate(self.player_colours):
                axials = self.board.axial_of_colour(colour)
                self.pins[player] = [
                    Pin(self.board, axials[i], id=i, color=colour)
                    for i in range(self.N_PINS)
                ]

        self._record_current_state()
        return self._get_obs(), self._info()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        legal_mask = self.action_masks()
        if action < 0 or action >= self.ACTION_DIM or not legal_mask[action]:
            return self._get_obs(), -0.5, False, False, {
                "illegal": True,
                **self._info(action_mask=legal_mask),
            }

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
                    **self._info(),
                }

            self._switch_turn()
            return self._get_obs(), reward, False, False, self._info()

        pin_id, direction_idx, is_jump = self._decode_action(action)
        pin = self.pins[self.current_player][pin_id]
        src_idx = pin.axialindex
        dest_idx = self._destination_for_submove(src_idx, direction_idx, is_jump)

        self.step_count += 1

        target_colour = self.board.colour_opposites[self.player_colours[self.current_player]]
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

        reward = -0.01
        reward += (dist_before - dist_after) * 0.1

        if src_idx not in target_index_set and dest_idx in target_index_set:
            reward += 0.25
        elif src_idx in target_index_set and dest_idx not in target_index_set:
            reward -= 0.25

        if is_jump:
            if self.active_jump_pin_id is None:
                self.active_jump_pin_id = pin_id
                self.jump_visited_cells = {src_idx, dest_idx}
            else:
                self.jump_visited_cells.add(dest_idx)

        visits = self._record_current_state()
        if visits > 1:
            reward -= 0.05 * (visits - 1)

        if self._has_player_won(self.current_player):
            self.done = True
            return self._get_obs(), 10.0, True, False, {
                "winner": self.current_player,
                **self._info(),
            }

        if self.step_count >= self.max_steps:
            self.done = True
            return self._get_obs(), reward, False, True, {
                "reason": "max_steps",
                **self._info(),
            }

        if not is_jump:
            self._switch_turn()
            return self._get_obs(), reward, False, False, self._info()

        if not self._has_continuing_jump(pin_id):
            self._switch_turn()

        return self._get_obs(), reward, False, False, self._info()

    def _get_obs(self) -> np.ndarray:
        l1 = np.zeros(self.N_CELLS, dtype=np.float32)
        for p in self.pins[self.current_player]:
            l1[p.axialindex] = 1.0

        l2 = np.zeros(self.N_CELLS, dtype=np.float32)
        for player in range(self.num_players):
            if player == self.current_player:
                continue
            for p in self.pins[player]:
                l2[p.axialindex] = 1.0

        l3 = np.zeros(self.N_CELLS, dtype=np.float32)
        target_colour = self.board.colour_opposites[self.player_colours[self.current_player]]
        for idx in self.board.axial_of_colour(target_colour):
            l3[idx] = 1.0

        l4 = np.zeros(self.N_CELLS, dtype=np.float32)
        if self.active_jump_pin_id is not None:
            active_pin = self.pins[self.current_player][self.active_jump_pin_id]
            l4[active_pin.axialindex] = 1.0

        return np.concatenate([l1, l2, l3, l4]).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.ACTION_DIM, dtype=bool)

        if self.board is None:
            mask[:] = True
            return mask

        if self.active_jump_pin_id is not None:
            pin = self.pins[self.current_player][self.active_jump_pin_id]
            for direction_idx in range(len(self.DIRECTIONS)):
                if self._is_legal_submove(pin, direction_idx, is_jump=True):
                    mask[self._encode_action(pin.id, direction_idx, True)] = True
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
        mult = 2 if is_jump else 1
        return self.board.index_of.get((src_cell.q + dq * mult, src_cell.r + dr * mult))

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
        self.current_player = (self.current_player + 1) % self.num_players

    def _info(self, action_mask=None):
        return {
            "action_mask": self.action_masks() if action_mask is None else action_mask,
            "current_player": self.current_player,
            "current_colour": self.player_colours[self.current_player],
            "num_players": self.num_players,
        }

    def _state_key(self):
        all_positions = tuple(
            tuple(sorted(p.axialindex for p in self.pins[player]))
            for player in range(self.num_players)
        )
        jump_state = (self.active_jump_pin_id, tuple(sorted(self.jump_visited_cells)))
        return (self.current_player, all_positions, jump_state)

    def _record_current_state(self):
        key = self._state_key()
        self.state_visit_counts[key] = self.state_visit_counts.get(key, 0) + 1
        return self.state_visit_counts[key]

    def _has_player_won(self, player: int) -> bool:
        target_colour = self.board.colour_opposites[self.player_colours[player]]
        target_cells = set(self.board.axial_of_colour(target_colour))
        current_positions = {p.axialindex for p in self.pins[player]}
        return current_positions == target_cells

    def render(self):
        if self.render_mode == "ansi" and self.board:
            all_pins = []
            for player_pins in self.pins:
                all_pins.extend(player_pins)
            self.board.print_ascii(pins=all_pins)
