import argparse
import os
import sys
import time


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from alpha_zero_prototype import AlphaZeroNet, AlphaZeroGuide, root_action_probs
from neural_guided_mcts import NeuralGuidedMCTS, SearchState


def watch_game(checkpoint_path, simulations=30, delay=0.35, max_steps=120):
    nnet = AlphaZeroNet()
    folder = os.path.dirname(checkpoint_path) or "."
    filename = os.path.basename(checkpoint_path)
    nnet.load_checkpoint(folder, filename)

    guide = AlphaZeroGuide(nnet)
    state = SearchState.new_game(max_steps=max_steps)

    print(f"Watching AlphaZero-style prototype game...")
    print(f"Loaded checkpoint: {checkpoint_path}")
    time.sleep(1.0)

    while not state.is_terminal():
        os.system("clear")
        print(f"Step {state.step_count} | Current player: {state.current_player}")
        state.render()

        searcher = NeuralGuidedMCTS(guide=guide, simulations=simulations, cpuct=1.5)
        action, root = searcher.run(state)
        if action is None:
            print("No legal actions available.")
            break

        pi = root_action_probs(root, state.action_dim, temperature=1e-6)
        action_idx = int(pi.argmax())
        chosen_action = state._decode_action(action_idx)

        print()
        print(f"Chosen action: {chosen_action.to_text()}")
        time.sleep(delay)

        state = state.apply_action(chosen_action)

    os.system("clear")
    print(f"Game finished after {state.step_count} steps")
    state.render()
    if state.winner is not None:
        print(f"Winner: player {state.winner}")
    else:
        print("No winner recorded.")


def main():
    parser = argparse.ArgumentParser(description="Watch a saved AlphaZero-style prototype checkpoint play.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="MCTS/azero_checkpoints/azero_submoves.pt",
        help="Path to a saved prototype checkpoint.",
    )
    parser.add_argument("--sims", type=int, default=30, help="MCTS simulations per move.")
    parser.add_argument("--delay", type=float, default=0.35, help="Delay between moves.")
    parser.add_argument("--max-steps", type=int, default=120, help="Maximum steps per game.")
    args = parser.parse_args()

    watch_game(
        checkpoint_path=args.checkpoint,
        simulations=args.sims,
        delay=args.delay,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
