import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


TEACHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "single system"))
if TEACHER_DIR not in sys.path:
    sys.path.append(TEACHER_DIR)

from checkers_board import HexBoard
from checkers_pins import Pin


def env(**kwargs):
    environment = raw_env(**kwargs)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


class raw_env(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "apo_chinese_checkers_marl"}

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

    def __init__(self, num_players=4, render_mode="ansi", max_steps=400, **kwargs):
        super().__init__()
        if num_players not in (2, 3, 4, 6):
            raise ValueError("num_players must be one of: 2, 3, 4, 6")

        self.num_players = num_players
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.player_colours = list(self.PLAYER_COLOUR_SETS[num_players])

        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }
        self._agent_selector = agent_selector.agent_selector(self.agents)

        with redirect_stdout(io.StringIO()):
            temp_board = HexBoard(R=4, hole_radius=16, spacing=34)
        self.N_CELLS = len(temp_board.cells)

        self.ACTIONS_PER_PIN = len(self.DIRECTIONS) * 2
        self.END_TURN_ACTION = self.N_PINS * self.ACTIONS_PER_PIN
        self.ACTION_DIM = self.END_TURN_ACTION + 1

        obs_dim = self.N_CELLS * 4
        self.action_spaces = {
            agent: Discrete(self.ACTION_DIM) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Dict(
                {
                    "observation": Box(
                        low=0.0,
                        high=1.0,
                        shape=(obs_dim,),
                        dtype=np.float32,
                    ),
                    "action_mask": Box(
                        low=0,
                        high=1,
                        shape=(self.ACTION_DIM,),
                        dtype=np.int8,
                    ),
                }
            )
            for agent in self.possible_agents
        }

        self.board = None
        self.pins = []
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()
        self.step_count = 0
        self.winner = None
        self.agent_selection = None

        self.rewards = None
        self._cumulative_rewards = None
        self.terminations = None
        self.truncations = None
        self.infos = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        with redirect_stdout(io.StringIO()):
            self.board = HexBoard(R=4, hole_radius=16, spacing=34)

        self.pins = [[] for _ in range(self.num_players)]
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()
        self.step_count = 0
        self.winner = None

        with redirect_stdout(io.StringIO()):
            for player, colour in enumerate(self.player_colours):
                axials = self.board.axial_of_colour(colour)
                self.pins[player] = [
                    Pin(self.board, axials[i], id=i, color=colour)
                    for i in range(self.N_PINS)
                ]

        self._agent_selector = agent_selector.agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent):
        player = self.agent_name_mapping[agent]
        observation = self._get_obs(player)
        action_mask = self._action_masks_for_player(player).astype(np.int8)
        return {"observation": observation, "action_mask": action_mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player = self.agent_name_mapping[agent]
        self._cumulative_rewards[agent] = 0.0
        self.rewards = {a: 0.0 for a in self.agents}

        legal_mask = self._action_masks_for_player(player)
        action = int(action)

        if action < 0 or action >= self.ACTION_DIM or not legal_mask[action]:
            self.rewards[agent] = -0.5
            self._accumulate_rewards()
            self._advance_turn()
            return

        if action == self.END_TURN_ACTION:
            self.step_count += 1
            self.rewards[agent] = -0.02
            self._advance_turn()
            self._handle_truncation()
            self._accumulate_rewards()
            return

        pin_id, direction_idx, is_jump = self._decode_action(action)
        pin = self.pins[player][pin_id]
        src_idx = pin.axialindex
        dest_idx = self._destination_for_submove(src_idx, direction_idx, is_jump)

        target_colour = self.board.colour_opposites[self.player_colours[player]]
        target_indices = self.board.axial_of_colour(target_colour)
        target_index_set = set(target_indices)
        target_cells = [self.board.cells[i] for i in target_indices]
        goal_q = sum(c.q for c in target_cells) / len(target_cells)
        goal_r = sum(c.r for c in target_cells) / len(target_cells)

        def get_dist_to_centroid(idx):
            c = self.board.cells[idx]
            return (
                abs(c.q - goal_q)
                + abs(c.q + c.r - goal_q - goal_r)
                + abs(c.r - goal_r)
            ) / 2

        dist_before = get_dist_to_centroid(src_idx)
        with redirect_stdout(io.StringIO()):
            pin.placePin(dest_idx)
        dist_after = get_dist_to_centroid(dest_idx)

        self.step_count += 1
        reward = -0.01 + (dist_before - dist_after) * 0.1

        if src_idx not in target_index_set and dest_idx in target_index_set:
            reward += 0.25
        elif src_idx in target_index_set and dest_idx not in target_index_set:
            reward -= 0.25

        self.rewards[agent] = reward

        if is_jump:
            if self.active_jump_pin_id is None:
                self.active_jump_pin_id = pin_id
                self.jump_visited_cells = {src_idx, dest_idx}
            else:
                self.jump_visited_cells.add(dest_idx)

        if self._has_player_won(player):
            self.winner = player
            for other_agent in self.agents:
                self.terminations[other_agent] = True
                self.rewards[other_agent] = 10.0 if other_agent == agent else -1.0
                self.infos[other_agent]["winner"] = player
        elif not is_jump or not self._has_continuing_jump(player, pin_id):
            self._advance_turn()

        self._handle_truncation()
        self._accumulate_rewards()

    def render(self):
        if self.render_mode == "ansi" and self.board:
            all_pins = []
            for player_pins in self.pins:
                all_pins.extend(player_pins)
            self.board.print_ascii(pins=all_pins)

    def close(self):
        pass

    def _get_obs(self, player):
        l1 = np.zeros(self.N_CELLS, dtype=np.float32)
        for pin in self.pins[player]:
            l1[pin.axialindex] = 1.0

        l2 = np.zeros(self.N_CELLS, dtype=np.float32)
        for other_player in range(self.num_players):
            if other_player == player:
                continue
            for pin in self.pins[other_player]:
                l2[pin.axialindex] = 1.0

        l3 = np.zeros(self.N_CELLS, dtype=np.float32)
        target_colour = self.board.colour_opposites[self.player_colours[player]]
        for idx in self.board.axial_of_colour(target_colour):
            l3[idx] = 1.0

        l4 = np.zeros(self.N_CELLS, dtype=np.float32)
        current_agent = self.agent_name_mapping[self.agent_selection] if self.agent_selection else 0
        if player == current_agent and self.active_jump_pin_id is not None:
            active_pin = self.pins[player][self.active_jump_pin_id]
            l4[active_pin.axialindex] = 1.0

        return np.concatenate([l1, l2, l3, l4]).astype(np.float32)

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

    def _is_legal_submove(self, player, pin, direction_idx, is_jump):
        current_player = self.agent_name_mapping[self.agent_selection]
        if player != current_player:
            return False
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

    def _action_masks_for_player(self, player):
        mask = np.zeros(self.ACTION_DIM, dtype=bool)
        current_player = self.agent_name_mapping[self.agent_selection]
        if player != current_player or self.board is None:
            return mask

        if self.active_jump_pin_id is not None:
            pin = self.pins[player][self.active_jump_pin_id]
            for direction_idx in range(len(self.DIRECTIONS)):
                if self._is_legal_submove(player, pin, direction_idx, is_jump=True):
                    mask[self._encode_action(pin.id, direction_idx, True)] = True
            mask[self.END_TURN_ACTION] = True
            return mask

        has_any_move = False
        for pin in self.pins[player]:
            for direction_idx in range(len(self.DIRECTIONS)):
                if self._is_legal_submove(player, pin, direction_idx, is_jump=False):
                    mask[self._encode_action(pin.id, direction_idx, False)] = True
                    has_any_move = True
                if self._is_legal_submove(player, pin, direction_idx, is_jump=True):
                    mask[self._encode_action(pin.id, direction_idx, True)] = True
                    has_any_move = True

        if not has_any_move:
            mask[self.END_TURN_ACTION] = True
        return mask

    def _has_continuing_jump(self, player, pin_id):
        pin = self.pins[player][pin_id]
        for direction_idx in range(len(self.DIRECTIONS)):
            if self._is_legal_submove(player, pin, direction_idx, is_jump=True):
                return True
        return False

    def _advance_turn(self):
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()
        self.agent_selection = self._agent_selector.next()

    def _has_player_won(self, player):
        target_colour = self.board.colour_opposites[self.player_colours[player]]
        target_cells = set(self.board.axial_of_colour(target_colour))
        current_positions = {pin.axialindex for pin in self.pins[player]}
        return current_positions == target_cells

    def _handle_truncation(self):
        if self.step_count >= self.max_steps and self.winner is None:
            for agent in self.agents:
                self.truncations[agent] = True
