import argparse
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


CURRENT_DIR = Path(__file__).resolve().parent
OLD_DIR = CURRENT_DIR / "old"
if str(OLD_DIR) not in sys.path:
    sys.path.append(str(OLD_DIR))

from APO_env_v2 import ChineseCheckersEnv


class TrainingLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_lens = []
        self.ep_rews = []
        self.completion_rates = []
        self._recent_outcomes = []
        self.window = 50

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.ep_lens.append(info["episode"]["l"])
                self.ep_rews.append(info["episode"]["r"])
                won = 1 if "winner" in info else 0
                self._recent_outcomes.append(won)
                recent = self._recent_outcomes[-self.window:]
                completion_rate = sum(recent) / len(recent)
                self.completion_rates.append(completion_rate)
                self.logger.record("rollout/game_completion_rate", completion_rate)
        return True

    def save_plot(self, output_path: Path):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        def smooth(y, window=20):
            if len(y) < window:
                return y
            return np.convolve(y, np.ones(window) / window, mode="valid")

        axes[0].plot(smooth(self.completion_rates, 10), color="blue")
        axes[0].set_title("Game Completion Rate")
        axes[0].set_xlabel("Episodes")
        axes[0].set_ylabel("Rate")
        axes[0].set_ylim(0, 1)

        axes[1].plot(smooth(self.ep_lens), color="purple")
        axes[1].set_title("Avg Episode Length")
        axes[1].set_xlabel("Episodes")
        axes[1].set_ylabel("Steps")

        axes[2].plot(smooth(self.ep_rews), color="cyan")
        axes[2].set_title("Avg Episode Reward")
        axes[2].set_xlabel("Episodes")
        axes[2].set_ylabel("Reward")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)


class MetricsCSVLogger(BaseCallback):
    def __init__(self, csv_path: Path, run_name: str, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.run_name = run_name
        self.iteration = 0
        self.fieldnames = [
            "run_name",
            "iteration",
            "total_timesteps",
            "time_elapsed",
            "fps",
            "completion_rate",
            "ep_len_mean",
            "ep_rew_mean",
            "approx_kl",
            "clip_fraction",
            "entropy_loss",
            "explained_variance",
            "learning_rate",
            "loss",
            "policy_gradient_loss",
            "value_loss",
        ]

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        values = self.model.logger.name_to_value
        row = {
            "run_name": self.run_name,
            "iteration": self.iteration,
            "total_timesteps": values.get("time/total_timesteps"),
            "time_elapsed": values.get("time/time_elapsed"),
            "fps": values.get("time/fps"),
            "completion_rate": values.get("rollout/game_completion_rate"),
            "ep_len_mean": values.get("rollout/ep_len_mean"),
            "ep_rew_mean": values.get("rollout/ep_rew_mean"),
            "approx_kl": values.get("train/approx_kl"),
            "clip_fraction": values.get("train/clip_fraction"),
            "entropy_loss": values.get("train/entropy_loss"),
            "explained_variance": values.get("train/explained_variance"),
            "learning_rate": values.get("train/learning_rate"),
            "loss": values.get("train/loss"),
            "policy_gradient_loss": values.get("train/policy_gradient_loss"),
            "value_loss": values.get("train/value_loss"),
        }
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def _on_step(self) -> bool:
        return True


class DeadlineStopper(BaseCallback):
    def __init__(self, deadline_ts: float, verbose=0):
        super().__init__(verbose)
        self.deadline_ts = deadline_ts

    def _on_step(self) -> bool:
        return time.time() < self.deadline_ts


def mask_fn(env):
    return env.unwrapped.action_masks()


def latest_checkpoint(checkpoint_dir: Path):
    checkpoints = sorted(checkpoint_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def build_env(max_steps: int):
    base_env = ChineseCheckersEnv(max_steps=max_steps)
    base_env = Monitor(base_env)
    return ActionMasker(base_env, mask_fn)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Long-running resumable PPO trainer for 1v1 submoves v2."
    )
    parser.add_argument("--run-name", default=os.environ.get("APO_RUN_NAME", "ppo_submoves_v2"))
    parser.add_argument("--hours", type=float, default=float(os.environ.get("APO_TRAIN_HOURS", "1")))
    parser.add_argument("--save-freq", type=int, default=int(os.environ.get("APO_SAVE_FREQ", "50000")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("APO_MAX_STEPS", "300")))
    parser.add_argument("--total-timesteps", type=int, default=int(os.environ.get("APO_TOTAL_TIMESTEPS", "5000000")))
    parser.add_argument("--resume", choices=["auto", "never"], default=os.environ.get("APO_RESUME", "auto"))
    parser.add_argument(
        "--init-model-path",
        default=os.environ.get("APO_INIT_MODEL_PATH"),
        help="Optional PPO zip to use as the starting model for a fresh run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_root = CURRENT_DIR.parent / "longrun_training" / args.run_name
    checkpoint_dir = run_root / "checkpoints"
    tensorboard_dir = run_root / "tensorboard"
    metrics_csv = run_root / "metrics.csv"
    plot_path = run_root / "training_curves.png"
    final_model_path = run_root / "final_model"
    latest_model_path = run_root / "latest_model"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args.max_steps)
    logger = TrainingLogger()
    metrics_logger = MetricsCSVLogger(metrics_csv, run_name=args.run_name)
    deadline_ts = time.time() + args.hours * 3600
    deadline_cb = DeadlineStopper(deadline_ts)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="checkers_ppo_submoves_v2",
        verbose=1,
    )

    resume_path = latest_checkpoint(checkpoint_dir) if args.resume == "auto" else None
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        model = MaskablePPO.load(
            str(resume_path),
            env=env,
            tensorboard_log=str(tensorboard_dir),
        )
    elif args.init_model_path:
        print(f"Starting from initial model: {args.init_model_path}")
        model = MaskablePPO.load(
            str(args.init_model_path),
            env=env,
            tensorboard_log=str(tensorboard_dir),
        )
    else:
        print("Starting fresh training run.")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            learning_rate=1e-4,
            clip_range=0.1,
            target_kl=0.02,
            max_grad_norm=0.5,
            ent_coef=0.005,
            batch_size=64,
            n_steps=1024,
        )

    print(f"Run directory: {run_root}")
    print(f"Training budget: {args.hours:.2f} hours")
    print(f"Checkpoint frequency: every {args.save_freq} timesteps")
    print(f"Max training timesteps cap: {args.total_timesteps}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[logger, checkpoint_callback, metrics_logger, deadline_cb],
        reset_num_timesteps=False,
    )

    model.save(str(final_model_path))
    model.save(str(latest_model_path))
    logger.save_plot(plot_path)
    print(f"Saved final model to {final_model_path}.zip")
    print(f"Saved latest model to {latest_model_path}.zip")
    print(f"Saved metrics to {metrics_csv}")
    print(f"Saved plots to {plot_path}")


if __name__ == "__main__":
    main()
