# MCTS Notes

This folder now contains a few separate search experiments built on top of the
teacher Chinese Checkers code.

## Files

- `basic_mcts.py`
  - Simple 1v1 MCTS baseline using the teacher move generator directly.
  - Run:
    - `python "MCTS/basic_mcts.py"`

- `neural_guided_mcts.py`
  - 1v1 submove-based MCTS that can use a trained PPO model for policy priors
    and value estimates during search.
  - Run without a PPO model:
    - `python "MCTS/neural_guided_mcts.py" --simulations 120`
  - Run with a PPO model:
    - `python "MCTS/neural_guided_mcts.py" --model-path "checkers_ppo_final.zip" --simulations 120`

- `alpha_zero_prototype.py`
  - Lightweight AlphaZero-style prototype.
  - Includes:
    - self-play
    - MCTS-informed policy targets
    - value targets from final outcomes
    - simple old-vs-new arena comparison
  - Run a tiny quick test:
    - `python "MCTS/alpha_zero_prototype.py" --iters 1 --episodes 1 --sims 3 --arena-games 2 --max-steps 20`
  - Run a larger prototype test:
    - `python "MCTS/alpha_zero_prototype.py" --iters 3 --episodes 5 --sims 25 --arena-games 4 --max-steps 80`

- `watch_alpha_zero.py`
  - Loads a saved AlphaZero-style prototype checkpoint and watches it play in
    the terminal.
  - Default checkpoint:
    - `MCTS/azero_checkpoints/azero_submoves.pt`
  - Run:
    - `python "MCTS/watch_alpha_zero.py"`

## Checkpoints

- AlphaZero-style prototype checkpoints are saved in:
  - `MCTS/azero_checkpoints/`

## Notes

- These files do not modify the teacher files in `single system/`.
- `basic_mcts.py` is the simplest baseline.
- `neural_guided_mcts.py` is useful for testing search guided by an existing PPO model.
- `alpha_zero_prototype.py` is a prototype, not a polished full AlphaZero system.
