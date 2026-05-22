import argparse
import copy
import os
import random
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from neural_guided_mcts import NeuralGuidedMCTS, SearchState


@dataclass
class AlphaZeroArgs:
    num_iters: int = 2
    num_episodes: int = 4
    num_mcts_sims: int = 30
    cpuct: float = 1.5
    max_steps: int = 120
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 1e-3
    arena_games: int = 4
    update_threshold: float = 0.55
    temperature_steps: int = 10
    parallel_games: int = 1
    history_iters: int = 5
    min_decisive_games: int = 2
    load_existing: bool = False
    checkpoint_dir: str = "MCTS/azero_checkpoints"
    model_name: str = "azero_submoves.pt"


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features))
        return logits, value


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ResidualPolicyValueNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        in_channels: int = 4,
        trunk_channels: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        board_area = input_dim // in_channels
        board_size = int(round(board_area ** 0.5))
        if board_size * board_size * in_channels != input_dim:
            raise ValueError(
                "ResidualPolicyValueNet expects a square board observation with "
                f"{in_channels} channels; got input_dim={input_dim}."
            )

        self.in_channels = in_channels
        self.board_size = board_size

        self.input_conv = nn.Conv2d(
            in_channels, trunk_channels, kernel_size=3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(trunk_channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(trunk_channels) for _ in range(num_blocks)]
        )

        self.policy_conv = nn.Conv2d(trunk_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_head = nn.Linear(2 * board_size * board_size, action_dim)

        self.value_conv = nn.Conv2d(trunk_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.in_channels, self.board_size, self.board_size)
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(batch_size, -1)
        logits = self.policy_head(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return logits, value


class AlphaZeroNet:
    def __init__(self, input_dim=None, action_dim=None, device=None, network_type="rescnn"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.network_type = network_type
        if input_dim is not None and action_dim is not None:
            self._build_model()

    def _build_model(self):
        if self.network_type == "mlp":
            self.model = PolicyValueNet(self.input_dim, self.action_dim).to(self.device)
        else:
            self.model = ResidualPolicyValueNet(self.input_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def ensure_shape(self, input_dim, action_dim):
        if self.model is None:
            self.input_dim = input_dim
            self.action_dim = action_dim
            self._build_model()

    def predict(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        self.ensure_shape(obs.shape[0], SearchState.new_game().action_dim)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            value = float(value.squeeze(0).squeeze(0).cpu().item())
        return probs, value

    def train(self, examples, args: AlphaZeroArgs):
        if not examples:
            return

        observations = np.array([ex[0] for ex in examples], dtype=np.float32)
        policies = np.array([ex[1] for ex in examples], dtype=np.float32)
        values = np.array([ex[2] for ex in examples], dtype=np.float32)
        self.ensure_shape(observations.shape[1], policies.shape[1])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        dataset_size = len(examples)
        self.model.train()

        for _ in range(args.epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, args.batch_size):
                batch_idx = indices[start:start + args.batch_size]
                obs_batch = torch.tensor(observations[batch_idx], dtype=torch.float32, device=self.device)
                pi_batch = torch.tensor(policies[batch_idx], dtype=torch.float32, device=self.device)
                v_batch = torch.tensor(values[batch_idx], dtype=torch.float32, device=self.device)

                logits, pred_v = self.model(obs_batch)
                log_probs = torch.log_softmax(logits, dim=-1)
                policy_loss = -(pi_batch * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_v.squeeze(-1), v_batch)
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def clone(self):
        clone = AlphaZeroNet(
            self.input_dim,
            self.action_dim,
            device=self.device,
            network_type=self.network_type,
        )
        clone.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        return clone

    def save_checkpoint(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "action_dim": self.action_dim,
                "network_type": self.network_type,
            },
            os.path.join(folder, filename),
        )

    def load_checkpoint(self, folder, filename):
        checkpoint = torch.load(os.path.join(folder, filename), map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.action_dim = checkpoint["action_dim"]
        self.network_type = checkpoint.get("network_type", "mlp")
        self._build_model()
        self.model.load_state_dict(checkpoint["state_dict"])


class AlphaZeroGuide:
    def __init__(self, nnet: AlphaZeroNet):
        self.nnet = nnet

    def evaluate(self, state: SearchState):
        legal_mask = state.legal_action_mask()
        if not legal_mask.any():
            return np.zeros(state.action_dim, dtype=np.float32), state.terminal_value()

        probs, value = self.nnet.predict(state.observe())
        priors = np.zeros_like(probs, dtype=np.float32)
        priors[legal_mask] = probs[legal_mask]
        total = priors.sum()
        if total <= 0:
            priors = legal_mask.astype(np.float32)
            priors /= priors.sum()
        else:
            priors /= total
        return priors, float(np.clip(value, -1.0, 1.0))


def root_action_probs(root, action_dim, temperature=1.0):
    visits = np.zeros(action_dim, dtype=np.float32)
    for action, child in root.children.items():
        if action.is_end_turn:
            action_idx = root.state.end_turn_action
        else:
            action_idx = root.state._encode_action(action.pin_id, action.direction_idx, action.is_jump)
        visits[action_idx] = child.visits

    if visits.sum() <= 0:
        return np.ones(action_dim, dtype=np.float32) / action_dim

    if temperature <= 1e-6:
        probs = np.zeros_like(visits)
        probs[np.argmax(visits)] = 1.0
        return probs

    adjusted = np.power(visits, 1.0 / temperature)
    total = adjusted.sum()
    if total <= 0:
        probs = np.zeros_like(visits)
        probs[np.argmax(visits)] = 1.0
        return probs
    return adjusted / total


class AlphaZeroCoach:
    def __init__(self, args: AlphaZeroArgs):
        self.args = args
        initial_state = SearchState.new_game(max_steps=args.max_steps)
        self.nnet = AlphaZeroNet(
            input_dim=initial_state.observe().shape[0],
            action_dim=initial_state.action_dim,
        )
        self.example_history = deque(maxlen=args.history_iters)

        checkpoint_path = os.path.join(args.checkpoint_dir, args.model_name)
        if args.load_existing and os.path.exists(checkpoint_path):
            self.nnet.load_checkpoint(args.checkpoint_dir, args.model_name)
            print(f"Loaded existing checkpoint: {checkpoint_path}")

    def _finalize_examples(self, examples, terminal_state):
        final_examples = []
        for observation, pi, player in examples:
            if terminal_state.winner is None:
                z = 0.0
            else:
                z = 1.0 if player == terminal_state.winner else -1.0
            final_examples.append((observation, pi, z))
        return final_examples

    def execute_episode(self):
        return self.execute_parallel_episodes(1)[0]

    def execute_parallel_episodes(self, num_games):
        guide = AlphaZeroGuide(self.nnet)
        games = []
        for _ in range(num_games):
            games.append(
                {
                    "state": SearchState.new_game(max_steps=self.args.max_steps),
                    "examples": [],
                }
            )

        finished_examples = []
        while games:
            next_games = []
            for game in games:
                state = game["state"]
                if state.is_terminal():
                    finished_examples.append(
                        self._finalize_examples(game["examples"], state)
                    )
                    continue

                searcher = NeuralGuidedMCTS(
                    guide=guide,
                    simulations=self.args.num_mcts_sims,
                    cpuct=self.args.cpuct,
                )
                action, root = searcher.run(state)
                if action is None:
                    finished_examples.append(
                        self._finalize_examples(game["examples"], state)
                    )
                    continue

                temperature = 1.0 if state.step_count < self.args.temperature_steps else 1e-6
                pi = root_action_probs(root, state.action_dim, temperature=temperature)
                game["examples"].append((state.observe().copy(), pi, state.current_player))

                action_idx = np.random.choice(np.arange(state.action_dim), p=pi)
                sampled_action = state._decode_action(int(action_idx))
                next_state = state.apply_action(sampled_action)
                game["state"] = next_state

                if next_state.is_terminal():
                    finished_examples.append(
                        self._finalize_examples(game["examples"], next_state)
                    )
                else:
                    next_games.append(game)
            games = next_games

        return finished_examples

    def play_match(self, first_net: AlphaZeroNet, second_net: AlphaZeroNet):
        state = SearchState.new_game(max_steps=self.args.max_steps)
        guides = [AlphaZeroGuide(first_net), AlphaZeroGuide(second_net)]

        while not state.is_terminal():
            searcher = NeuralGuidedMCTS(
                guide=guides[state.current_player],
                simulations=self.args.num_mcts_sims,
                cpuct=self.args.cpuct,
            )
            action, root = searcher.run(state)
            if action is None:
                break
            pi = root_action_probs(root, state.action_dim, temperature=1e-6)
            action_idx = int(np.argmax(pi))
            state = state.apply_action(state._decode_action(action_idx))

        if state.winner is None:
            return 0
        return 1 if state.winner == 0 else -1

    def arena_compare(self, old_net: AlphaZeroNet, new_net: AlphaZeroNet):
        wins_new = 0
        wins_old = 0
        draws = 0

        for game_idx in range(self.args.arena_games):
            if game_idx % 2 == 0:
                result = self.play_match(old_net, new_net)
                if result == -1:
                    wins_new += 1
                elif result == 1:
                    wins_old += 1
                else:
                    draws += 1
            else:
                result = self.play_match(new_net, old_net)
                if result == 1:
                    wins_new += 1
                elif result == -1:
                    wins_old += 1
                else:
                    draws += 1

        total_decisive = wins_new + wins_old
        win_rate = wins_new / total_decisive if total_decisive > 0 else 0.0
        return wins_new, wins_old, draws, win_rate

    def learn(self):
        for iteration in range(1, self.args.num_iters + 1):
            print(f"------ AlphaZero Iter {iteration} ------")

            iteration_examples = []
            episode = 1
            while episode <= self.args.num_episodes:
                batch_games = min(self.args.parallel_games, self.args.num_episodes - episode + 1)
                batch_examples = self.execute_parallel_episodes(batch_games)
                for local_idx, episode_examples in enumerate(batch_examples, start=0):
                    iteration_examples.extend(episode_examples)
                    print(
                        f"  self-play episode {episode + local_idx}/{self.args.num_episodes} "
                        f"-> {len(episode_examples)} examples"
                    )
                episode += batch_games

            self.example_history.append(iteration_examples)
            examples = []
            for past_examples in self.example_history:
                examples.extend(past_examples)
            print(
                f"  replay history -> {len(self.example_history)} iteration sets, "
                f"{len(examples)} total examples"
            )

            old_net = self.nnet.clone()
            self.nnet.train(examples, self.args)

            wins_new, wins_old, draws, win_rate = self.arena_compare(old_net, self.nnet)
            print(
                f"  arena results -> new:{wins_new} old:{wins_old} draws:{draws} "
                f"win_rate:{win_rate:.3f}"
            )

            decisive_games = wins_new + wins_old
            if decisive_games < self.args.min_decisive_games:
                print("  rejecting new model, not enough decisive arena games")
                self.nnet = old_net
            elif win_rate < self.args.update_threshold:
                print("  rejecting new model, restoring previous checkpoint")
                self.nnet = old_net
            else:
                print("  accepting new model")
                self.nnet.save_checkpoint(self.args.checkpoint_dir, self.args.model_name)


def main():
    parser = argparse.ArgumentParser(description="Tiny AlphaZero-style prototype for Chinese Checkers.")
    parser.add_argument("--iters", type=int, default=2, help="Training iterations.")
    parser.add_argument("--episodes", type=int, default=4, help="Self-play episodes per iteration.")
    parser.add_argument("--sims", type=int, default=30, help="MCTS simulations per move.")
    parser.add_argument("--arena-games", type=int, default=4, help="Old-vs-new comparison games.")
    parser.add_argument("--max-steps", type=int, default=120, help="Maximum steps per game.")
    parser.add_argument("--parallel-games", type=int, default=1, help="How many self-play games to run in a round-robin batch.")
    parser.add_argument("--history-iters", type=int, default=5, help="How many past iteration example sets to keep.")
    parser.add_argument("--min-decisive-games", type=int, default=2, help="Minimum non-draw arena games before accepting a new model.")
    parser.add_argument("--load-existing", action="store_true", help="Load an existing checkpoint before continuing training.")
    parser.add_argument("--checkpoint-dir", type=str, default="MCTS/azero_checkpoints", help="Checkpoint directory.")
    parser.add_argument("--model-name", type=str, default="azero_submoves.pt", help="Checkpoint filename.")
    args_ns = parser.parse_args()

    args = AlphaZeroArgs(
        num_iters=args_ns.iters,
        num_episodes=args_ns.episodes,
        num_mcts_sims=args_ns.sims,
        arena_games=args_ns.arena_games,
        max_steps=args_ns.max_steps,
        parallel_games=max(1, args_ns.parallel_games),
        history_iters=args_ns.history_iters,
        min_decisive_games=args_ns.min_decisive_games,
        load_existing=args_ns.load_existing,
        checkpoint_dir=args_ns.checkpoint_dir,
        model_name=args_ns.model_name,
    )
    coach = AlphaZeroCoach(args)
    coach.learn()


if __name__ == "__main__":
    main()
