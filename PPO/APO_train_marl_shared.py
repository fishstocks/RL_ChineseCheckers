import csv
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from APO_env_marl import ChineseCheckersEnvMARL


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

    def plot(self):
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
        plt.savefig("training_curves_marl.png")
        plt.show()
        print("Saved to training_curves_marl.png")


class MetricsCSVLogger(BaseCallback):
    def __init__(self, csv_path, run_name, verbose=0):
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
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
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
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def _on_step(self) -> bool:
        return True


def mask_fn(env):
    return env.unwrapped.action_masks()


NUM_PLAYERS = int(os.environ.get("APO_NUM_PLAYERS", "4"))

base_env = ChineseCheckersEnvMARL(num_players=NUM_PLAYERS)
base_env = Monitor(base_env)
env = ActionMasker(base_env, mask_fn)

logger = TrainingLogger()
run_name = os.environ.get("APO_RUN_NAME", f"ppo_marl_{NUM_PLAYERS}p")
metrics_csv = os.path.join("training_logs", f"{run_name}.csv")
metrics_logger = MetricsCSVLogger(metrics_csv, run_name=run_name)

checkpoint_dir = "./checkpoints/"
tensorboard_dir = "./checkers_tensorboard/"

for path in [checkpoint_dir, tensorboard_dir]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=checkpoint_dir,
    name_prefix="checkers_ppo_marl",
    verbose=1,
)

print(f"Starting MARL-style shared-policy training with {NUM_PLAYERS} players...")
model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=tensorboard_dir,
    learning_rate=1e-4,
    clip_range=0.1,
    target_kl=0.02,
    max_grad_norm=0.5,
    ent_coef=0.005,
    batch_size=64,
    n_steps=1024,
)

model.learn(
    total_timesteps=1_000_000,
    callback=[logger, checkpoint_callback, metrics_logger],
    reset_num_timesteps=False,
)
model.save(f"checkers_ppo_marl_{NUM_PLAYERS}p")
logger.plot()
print(f"Saved metrics to {metrics_csv}")
