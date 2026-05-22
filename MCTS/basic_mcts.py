import argparse
import io
import math
import os
import random
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass


TEACHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "single system"))
if TEACHER_DIR not in sys.path:
    sys.path.append(TEACHER_DIR)

from checkers_board import HexBoard
from checkers_pins import Pin


COLOURS = ["red", "blue"]


@dataclass(frozen=True)
class Action:
    pin_id: int
    dest_idx: int


class GameState:
    def __init__(self, board, pins, current_player=0, step_count=0, max_steps=150, winner=None):
        self.board = board
        self.pins = pins
        self.current_player = current_player
        self.step_count = step_count
        self.max_steps = max_steps
        self.winner = winner

    @classmethod
    def new_game(cls, max_steps=150):
        with redirect_stdout(io.StringIO()):
            board = HexBoard(R=4, hole_radius=16, spacing=34)
        pins = [[], []]
        with redirect_stdout(io.StringIO()):
            for player, colour in enumerate(COLOURS):
                axials = board.axial_of_colour(colour)
                pins[player] = [Pin(board, axials[i], id=i, color=colour) for i in range(10)]
        return cls(board=board, pins=pins, current_player=0, step_count=0, max_steps=max_steps, winner=None)

    def clone(self):
        with redirect_stdout(io.StringIO()):
            new_board = HexBoard(R=4, hole_radius=16, spacing=34)
        new_pins = [[], []]
        with redirect_stdout(io.StringIO()):
            for player in range(2):
                for pin in self.pins[player]:
                    new_pins[player].append(
                        Pin(new_board, pin.axialindex, id=pin.id, color=pin.color)
                    )
        return GameState(
            board=new_board,
            pins=new_pins,
            current_player=self.current_player,
            step_count=self.step_count,
            max_steps=self.max_steps,
            winner=self.winner,
        )

    def legal_actions(self):
        if self.is_terminal():
            return []
        actions = []
        for pin in self.pins[self.current_player]:
            for dest in pin.getPossibleMoves():
                actions.append(Action(pin.id, dest))
        return actions

    def apply_action(self, action):
        next_state = self.clone()
        if action is None:
            next_state.step_count += 1
            next_state.current_player = 1 - next_state.current_player
            return next_state

        pin = next_state.pins[next_state.current_player][action.pin_id]
        with redirect_stdout(io.StringIO()):
            pin.placePin(action.dest_idx)
        next_state.step_count += 1

        if next_state.has_player_won(next_state.current_player):
            next_state.winner = next_state.current_player
            return next_state

        next_state.current_player = 1 - next_state.current_player
        return next_state

    def is_terminal(self):
        return self.winner is not None or self.step_count >= self.max_steps

    def has_player_won(self, player):
        target_colour = self.board.colour_opposites[COLOURS[player]]
        target_cells = set(self.board.axial_of_colour(target_colour))
        current_positions = {p.axialindex for p in self.pins[player]}
        return current_positions == target_cells

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

    def evaluate_for_current_player(self):
        if self.winner is not None:
            return 1.0
        if self.step_count >= self.max_steps:
            return 0.0

        me = self.current_player
        opp = 1 - me
        my_dist = self.total_goal_distance(me)
        opp_dist = self.total_goal_distance(opp)

        my_target = self.board.axial_of_colour(self.board.colour_opposites[COLOURS[me]])
        opp_target = self.board.axial_of_colour(self.board.colour_opposites[COLOURS[opp]])
        my_home_count = sum(1 for p in self.pins[me] if p.axialindex in my_target)
        opp_home_count = sum(1 for p in self.pins[opp] if p.axialindex in opp_target)

        distance_score = (opp_dist - my_dist) / 100.0
        home_score = (my_home_count - opp_home_count) / 10.0
        return max(-1.0, min(1.0, distance_score + home_score))

    def action_to_text(self, action):
        if action is None:
            return "PASS"
        src = self.pins[self.current_player][action.pin_id].axialindex
        return f"pin {action.pin_id}: {src} -> {action.dest_idx}"


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = state.legal_actions()
        self.visits = 0
        self.total_value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        def score(child):
            exploitation = child.total_value / max(child.visits, 1)
            exploration = c_param * math.sqrt(math.log(max(self.visits, 1)) / max(child.visits, 1))
            return exploitation + exploration

        return max(self.children, key=score)

    def expand(self, state):
        action = self.untried_actions.pop(random.randrange(len(self.untried_actions)))
        child_state = state.apply_action(action)
        child = Node(child_state, parent=self, action=action)
        self.children.append(child)
        return child


class BasicMCTS:
    def __init__(self, simulations=100, rollout_depth=20, exploration=1.4):
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.exploration = exploration

    def search(self, root_state):
        root = Node(root_state)
        if not root.untried_actions:
            return None, []

        for _ in range(self.simulations):
            node = root
            state = root_state.clone()

            while not state.is_terminal() and node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration)
                state = state.apply_action(node.action)

            if not state.is_terminal() and node.untried_actions:
                node = node.expand(state)
                state = node.state

            value = self.rollout(state)

            while node is not None:
                node.visits += 1
                node.total_value += value
                value = -value
                node = node.parent

        ranked_children = sorted(
            root.children,
            key=lambda child: child.visits,
            reverse=True,
        )
        best = ranked_children[0]
        return best.action, ranked_children

    def rollout(self, state):
        rollout_state = state.clone()
        depth = 0

        while not rollout_state.is_terminal() and depth < self.rollout_depth:
            actions = rollout_state.legal_actions()
            if not actions:
                break

            action = self.greedy_rollout_action(rollout_state, actions)
            rollout_state = rollout_state.apply_action(action)
            depth += 1

        return rollout_state.evaluate_for_current_player()

    def greedy_rollout_action(self, state, actions):
        best_score = -float("inf")
        best_action = actions[0]

        for action in actions:
            next_state = state.apply_action(action)
            score = -next_state.evaluate_for_current_player()
            if score > best_score:
                best_score = score
                best_action = action

        return best_action


def print_top_actions(state, ranked_children, top_k=5):
    print("Top searched actions:")
    for child in ranked_children[:top_k]:
        avg_value = child.total_value / max(child.visits, 1)
        print(
            f"  {state.action_to_text(child.action)} | visits={child.visits} | avg_value={avg_value:.3f}"
        )


def play_game(simulations=120, delay=0.35, max_steps=150):
    state = GameState.new_game(max_steps=max_steps)
    searcher = BasicMCTS(simulations=simulations, rollout_depth=16, exploration=1.4)

    while not state.is_terminal():
        os.system("clear")
        print(f"Step {state.step_count} | Current player: {state.current_player} ({COLOURS[state.current_player]})")
        state.board.print_ascii(pins=state.pins[0] + state.pins[1])
        action, ranked_children = searcher.search(state)
        print()
        print_top_actions(state, ranked_children)
        print()
        print(f"Chosen action: {state.action_to_text(action)}")
        state = state.apply_action(action)
        time.sleep(delay)

    os.system("clear")
    print(f"Game finished after {state.step_count} steps")
    state.board.print_ascii(pins=state.pins[0] + state.pins[1])
    if state.winner is not None:
        print(f"Winner: player {state.winner} ({COLOURS[state.winner]})")
    else:
        print("Game ended because max steps were reached.")


def main():
    parser = argparse.ArgumentParser(description="Basic MCTS demo for 1v1 Chinese Checkers.")
    parser.add_argument("--simulations", type=int, default=120, help="Tree-search simulations per move.")
    parser.add_argument("--delay", type=float, default=0.35, help="Seconds to pause between moves.")
    parser.add_argument("--max-steps", type=int, default=150, help="Maximum moves before the demo stops.")
    args = parser.parse_args()

    play_game(simulations=args.simulations, delay=args.delay, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
