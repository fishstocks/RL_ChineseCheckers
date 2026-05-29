import io
import os
import sys
from contextlib import redirect_stdout
from functools import lru_cache

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------- Import original board code ---------------- #
# This PPO environment does not create the board rules from scratch.
# It reuses HexBoard and Pin from "single system", then wraps that
# board in a Gymnasium environment so MaskablePPO can train on it.
TEACHER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "single system")
)
if TEACHER_DIR not in sys.path:
    sys.path.append(TEACHER_DIR)

from checkers_board import HexBoard
from checkers_pins import Pin


# ---------------- Class: ChineseCheckersEnv ---------------- #
class ChineseCheckersEnv(gym.Env):
    """
    Two-player training environment for ppo_1mil_1v1.zip.

    Player 0 is red. Player 1 is blue. The same PPO policy is used for
    whichever player is currently active, so observations are written from
    the current player's point of view: "my pins", "enemy pins", "my goal".
    """
    metadata = {"render_modes": ["ansi"]}

    N_PINS = 10

    # Six neighboring directions on the axial hex grid.
    # A move chooses a direction and either steps 1 cell or jumps 2 cells.
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

        # Create a temporary board only to discover how many board cells exist.
        # In this board there are 121 cells.
        with redirect_stdout(io.StringIO()):
            temp_board = HexBoard(R=4, hole_radius=16, spacing=34)
        self.N_CELLS = len(temp_board.cells)

        # ---------------- Action space ---------------- #
        # PPO chooses one integer action:
        #   10 pins * 6 directions * 2 move types = 120 submove actions
        #   plus one END_TURN action = 121 total actions.
        self.ACTIONS_PER_PIN = len(self.DIRECTIONS) * 2
        self.END_TURN_ACTION = self.N_PINS * self.ACTIONS_PER_PIN
        self.ACTION_DIM = self.END_TURN_ACTION + 1
        self.action_space = spaces.Discrete(self.ACTION_DIM)

        # ---------------- Observation space ---------------- #
        # PPO sees four binary layers flattened into one vector:
        #   layer 1: current player's pins
        #   layer 2: opponent pins
        #   layer 3: current player's target triangle
        #   layer 4: active jump pin, if a multi-jump is in progress
        # 121 cells * 4 layers = 484 numbers.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.N_CELLS * 4,),
            dtype=np.float32,
        )

        # ---------------- Game state ---------------- #
        self.colours = ["red", "blue"]
        self.board = None
        self.pins = [[], []]
        self.current_player = 0
        self.done = True
        self.step_count = 0

        # Counts repeated board states so loops can be penalized.
        self.state_visit_counts = {}

        # Multi-jump tracking. If a pin jumps and can jump again, only that
        # same pin may continue until the jump chain ends.
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()

        # Tracks repeated pin movement, so PPO is discouraged from moving the
        # same piece again and again while ignoring the rest.
        self.last_moved_pin_id = [None, None]
        self.repeat_move_streak = [0, 0]

    def reset(self, *, seed=None, options=None):
        """Start a fresh episode and return observation + legal-action mask."""
        super().reset(seed=seed)

        # Fresh board for every episode.
        with redirect_stdout(io.StringIO()):
            self.board = HexBoard(R=4, hole_radius=16, spacing=34)

        # Reset all episode state.
        self.pins = [[], []]
        self.current_player = 0
        self.done = False
        self.step_count = 0
        self.state_visit_counts = {}
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()
        self.last_moved_pin_id = [None, None]
        self.repeat_move_streak = [0, 0]

        # Place 10 pins for each player in their starting triangles.
        with redirect_stdout(io.StringIO()):
            for player, colour in enumerate(self.colours):
                axials = self.board.axial_of_colour(colour)
                self.pins[player] = [
                    Pin(self.board, axials[i], id=i, color=colour)
                    for i in range(self.N_PINS)
                ]

        # Store the initial state so repeated positions can be detected later.
        self._record_current_state()
        return self._get_obs(), {"action_mask": self.action_masks()}

    def step(self, action: int):
        """
        Apply one submove action.

        Returns the Gymnasium tuple:
        observation, reward, terminated, truncated, info.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        # MaskablePPO should only choose legal actions, but this keeps the env
        # safe if called manually or with a different policy.
        legal_mask = self.action_masks()
        if action < 0 or action >= self.ACTION_DIM or not legal_mask[action]:
            return self._get_obs(), -0.5, False, False, {
                "illegal": True,
                "action_mask": legal_mask,
            }

        player = self.current_player

        # The goal is always the opposite colored triangle.
        target_colour = self.board.colour_opposites[self.colours[player]]
        target_indices = tuple(sorted(self.board.axial_of_colour(target_colour)))
        target_index_set = set(target_indices)

        # ---------------- End-turn action ---------------- #
        # This is mostly useful during jump chains. It receives a small penalty
        # so PPO prefers useful moves over ending turns too early.
        if action == self.END_TURN_ACTION:
            self.step_count += 1
            reward = -0.02

            if self.step_count >= self.max_steps:
                self.done = True
                visits = self._record_current_state()
                if visits > 1:
                    reward -= 0.05 * (visits - 1)
                return self._get_obs(), reward, False, True, {
                    "reason": "max_steps",
                    "action_mask": self.action_masks(),
                }

            self._switch_turn()
            visits = self._record_current_state()
            if visits > 1:
                reward -= 0.05 * (visits - 1)
            return self._get_obs(), reward, False, False, {"action_mask": self.action_masks()}

        # ---------------- Before-move measurements ---------------- #
        # These values let reward shaping compare whether the move improved
        # the player's position.
        positions_before = [pin.axialindex for pin in self.pins[player]]
        assignment_before = self._assignment_cost(positions_before, target_indices)
        top3_before = self._top_k_goal_distance_sum(positions_before, target_indices, k=3)
        home_before = sum(1 for pos in positions_before if pos in target_index_set)

        # Turn the PPO integer action into: which pin, which direction,
        # and whether it is a step or jump.
        pin_id, direction_idx, is_jump = self._decode_action(action)
        pin = self.pins[player][pin_id]
        src_idx = pin.axialindex
        dest_idx = self._destination_for_submove(src_idx, direction_idx, is_jump)

        # Actually move the pin using the original Pin logic.
        moved_piece_before = self._goal_distance(src_idx, target_indices)
        with redirect_stdout(io.StringIO()):
            pin.placePin(dest_idx)
        moved_piece_after = self._goal_distance(dest_idx, target_indices)

        # ---------------- After-move measurements ---------------- #
        positions_after = [pin.axialindex for pin in self.pins[player]]
        assignment_after = self._assignment_cost(positions_after, target_indices)
        top3_after = self._top_k_goal_distance_sum(positions_after, target_indices, k=3)
        home_after = sum(1 for pos in positions_after if pos in target_index_set)

        self.step_count += 1

        # ---------------- Reward shaping ---------------- #
        # This teaches PPO before it can reliably find full wins.
        reward = -0.01

        # Reward the moved pin for getting closer to the goal.
        reward += (moved_piece_before - moved_piece_after) * 0.06

        # Reward the whole team of pins for matching the target triangle better.
        reward += (assignment_before - assignment_after) * 0.12

        # Reward progress by the farthest-behind pins.
        reward += (top3_before - top3_after) * 0.08

        # Reward increasing the number of pins inside the goal triangle.
        reward += (home_after - home_before) * 0.35

        # Extra reward for entering the goal, extra penalty for leaving it.
        if src_idx not in target_index_set and dest_idx in target_index_set:
            reward += 0.35
        elif src_idx in target_index_set and dest_idx not in target_index_set:
            reward -= 0.45

        # Discourage moving the same pin repeatedly.
        if self.last_moved_pin_id[player] == pin_id:
            self.repeat_move_streak[player] += 1
        else:
            self.last_moved_pin_id[player] = pin_id
            self.repeat_move_streak[player] = 1

        if self.repeat_move_streak[player] >= 3:
            reward -= 0.03 * (self.repeat_move_streak[player] - 2)

        # ---------------- Jump-chain handling ---------------- #
        # A jump can continue with the same pin if more jumps are available.
        if is_jump:
            if self.active_jump_pin_id is None:
                self.active_jump_pin_id = pin_id
                self.jump_visited_cells = {src_idx, dest_idx}
            else:
                self.jump_visited_cells.add(dest_idx)

            if self._has_continuing_jump(pin_id):
                reward += 0.03

        # Win gives a large terminal reward.
        if self._has_player_won(player):
            self.done = True
            visits = self._record_current_state()
            if visits > 1:
                reward -= 0.05 * (visits - 1)
            return self._get_obs(), 20.0, True, False, {
                "winner": player,
                "action_mask": self.action_masks(),
            }

        # Stop episodes that run too long.
        if self.step_count >= self.max_steps:
            self.done = True
            visits = self._record_current_state()
            if visits > 1:
                reward -= 0.05 * (visits - 1)
            return self._get_obs(), reward, False, True, {
                "reason": "max_steps",
                "action_mask": self.action_masks(),
            }

        # A normal step always ends the turn.
        if not is_jump:
            self._switch_turn()
            visits = self._record_current_state()
            if visits > 1:
                reward -= 0.05 * (visits - 1)
            return self._get_obs(), reward, False, False, {"action_mask": self.action_masks()}

        # A jump ends the turn only if it cannot continue.
        if not self._has_continuing_jump(pin_id):
            self._switch_turn()
            visits = self._record_current_state()
            if visits > 1:
                reward -= 0.05 * (visits - 1)
            return self._get_obs(), reward, False, False, {"action_mask": self.action_masks()}

        # Otherwise the same player keeps moving the same jumping pin.
        visits = self._record_current_state()
        if visits > 1:
            reward -= 0.05 * (visits - 1)
        return self._get_obs(), reward, False, False, {"action_mask": self.action_masks()}

    def _goal_distance(self, idx, target_indices):
        """Distance from one cell to the center of the target triangle."""
        target_cells = [self.board.cells[i] for i in target_indices]
        goal_q = sum(c.q for c in target_cells) / len(target_cells)
        goal_r = sum(c.r for c in target_cells) / len(target_cells)
        c = self.board.cells[idx]
        return (
            abs(c.q - goal_q)
            + abs(c.q + c.r - goal_q - goal_r)
            + abs(c.r - goal_r)
        ) / 2

    def _hex_distance(self, src_idx, dst_idx):
        """True hex-grid distance between two board cells."""
        src = self.board.cells[src_idx]
        dst = self.board.cells[dst_idx]
        return (
            abs(src.q - dst.q)
            + abs((src.q + src.r) - (dst.q + dst.r))
            + abs(src.r - dst.r)
        ) / 2

    def _assignment_cost(self, positions, target_indices):
        """
        Measures how well all pins can be paired with target cells.

        Lower cost means the group of pins is closer to filling the goal
        triangle. This is used in the reward, not as a game rule.
        """
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
                dist = self._hex_distance(positions[i], target_idx)
                best = min(best, dist + dp(i + 1, used_mask | (1 << target_i)))
            return best

        return dp(0, 0)

    def _top_k_goal_distance_sum(self, positions, target_indices, k=3):
        """Sum distances for the k farthest pins, so lagging pins matter."""
        dists = sorted(
            (self._goal_distance(idx, target_indices) for idx in positions),
            reverse=True,
        )
        return float(sum(dists[:k]))

    def _get_obs(self) -> np.ndarray:
        """
        Build PPO's 484-number observation:
        l1 = current player's pins
        l2 = opponent pins
        l3 = current player's target triangle
        l4 = active jump pin, if a jump chain is happening
        """
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

    def action_masks(self) -> np.ndarray:
        """
        Build the legal-action mask for MaskablePPO.

        A True entry means the action id is legal right now.
        A False entry means PPO should not choose that action.
        """
        mask = np.zeros(self.ACTION_DIM, dtype=bool)

        if self.board is None:
            mask[:] = True
            return mask

        # If a multi-jump is active, only that pin can keep jumping.
        # END_TURN is also legal so the model can stop the chain.
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

        # If no jump chain is active, list all legal steps and jumps.
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
        """Convert pin + direction + step/jump into one PPO action number."""
        return pin_id * self.ACTIONS_PER_PIN + direction_idx * 2 + int(is_jump)

    def _decode_action(self, action):
        """Convert one PPO action number back into pin + direction + step/jump."""
        pin_id = action // self.ACTIONS_PER_PIN
        local = action % self.ACTIONS_PER_PIN
        direction_idx = local // 2
        is_jump = bool(local % 2)
        return pin_id, direction_idx, is_jump

    def _destination_for_submove(self, src_idx, direction_idx, is_jump):
        """Find the destination cell for one step or one jump."""
        src_cell = self.board.cells[src_idx]
        dq, dr = self.DIRECTIONS[direction_idx]
        multiplier = 2 if is_jump else 1
        q = src_cell.q + dq * multiplier
        r = src_cell.r + dr * multiplier
        return self.board.index_of.get((q, r))

    def _is_legal_submove(self, pin, direction_idx, is_jump):
        """Check if one step or one jump is legal for this pin."""
        if self.active_jump_pin_id is not None and pin.id != self.active_jump_pin_id:
            return False

        src_idx = pin.axialindex
        src_cell = self.board.cells[src_idx]
        dq, dr = self.DIRECTIONS[direction_idx]

        if is_jump:
            # Jump: adjacent cell must be occupied, landing cell must be empty.
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

        # Step: adjacent destination exists and is empty.
        if self.active_jump_pin_id is not None:
            return False

        dst_idx = self.board.index_of.get((src_cell.q + dq, src_cell.r + dr))
        if dst_idx is None:
            return False
        return not self.board.cells[dst_idx].occupied

    def _has_continuing_jump(self, pin_id):
        """Return True if the same pin can jump again."""
        pin = self.pins[self.current_player][pin_id]
        for direction_idx in range(len(self.DIRECTIONS)):
            if self._is_legal_submove(pin, direction_idx, is_jump=True):
                return True
        return False

    def _switch_turn(self):
        """End the current turn and clear jump-chain state."""
        self.active_jump_pin_id = None
        self.jump_visited_cells = set()
        self.current_player = 1 - self.current_player

    def _state_key(self):
        """Create a hashable snapshot of the board for loop detection."""
        red_positions = tuple(sorted(p.axialindex for p in self.pins[0]))
        blue_positions = tuple(sorted(p.axialindex for p in self.pins[1]))
        jump_state = (
            self.active_jump_pin_id,
            tuple(sorted(self.jump_visited_cells)),
        )
        return (self.current_player, red_positions, blue_positions, jump_state)

    def _record_current_state(self) -> int:
        """Record and return how many times this exact state has appeared."""
        key = self._state_key()
        self.state_visit_counts[key] = self.state_visit_counts.get(key, 0) + 1
        return self.state_visit_counts[key]

    def _has_player_won(self, player: int) -> bool:
        """A player wins when their pins exactly fill the opposite triangle."""
        target_colour = self.board.colour_opposites[self.colours[player]]
        target_cells = set(self.board.axial_of_colour(target_colour))
        current_positions = {p.axialindex for p in self.pins[player]}
        return current_positions == target_cells

    def render(self):
        """Print the board when render_mode is 'ansi'."""
        if self.render_mode == "ansi" and self.board:
            self.board.print_ascii(pins=self.pins[0] + self.pins[1])
