import time
import os
from sb3_contrib import MaskablePPO
from APO_env import ChineseCheckersEnv

model_path = "checkers_ppo_final.zip"
if not os.path.exists(model_path):
    checkpoint_dir = "checkpoints"
    if os.path.isdir(checkpoint_dir):
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
        if files:
            latest = max(files, key=lambda f: int(f.split("_steps")[0].split("_")[-1]))
            model_path = os.path.join(checkpoint_dir, latest)
        else:
            raise FileNotFoundError("No trained model found. Train first or create a checkpoint.")
    else:
        raise FileNotFoundError("No trained model found. Train first or create a checkpoint.")

env = ChineseCheckersEnv(render_mode="ansi")
model = MaskablePPO.load(model_path, env=env)

obs, info = env.reset()
done = False
step = 0

print("Watching one AI game...")
print(f"Loaded model: {model_path}")
time.sleep(1.0)

while not done:
    os.system("clear")
    print(f"Step {step} | Current player: {env.current_player}")
    env.render()
    action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
    print(f"Chosen action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step += 1
    time.sleep(0.3)

os.system("clear")
print(f"\nGame finished after {step} steps")
env.render()

if "winner" in info:
    print(f"Winner: player {info['winner']}")
elif truncated:
    print("Game ended because max steps were reached.")
else:
    print("Game ended without a recorded winner.")



# ---------------------------------------------------------
# Expected training output (Stable Baselines training log)
#
# At the start of training the agent has not learned yet.
# Episodes often hit the maximum number of steps and rewards
# are low.
#
# Example early training output:
#
# ---------------------------------
# | rollout/           |          |
# |    ep_len_mean     | 500      |  # hitting max_steps
# |    ep_rew_mean     | -0.2     |  # reward low at start
# | time/              |          |
# |    total_timesteps | 10000    |
# ---------------------------------
#
# As training progresses the agent begins to learn a better
# policy. Episodes finish faster and rewards increase.
#
# Example later training output:
#
# ---------------------------------
# | rollout/           |          |
# |    ep_len_mean     | 180      |  # finishing episodes faster
# |    ep_rew_mean     | 4.2      |  # reward improving
# ---------------------------------
#
# Lower episode length + higher reward usually indicates
# the agent is learning the task successfully.
# ---------------------------------------------------------
