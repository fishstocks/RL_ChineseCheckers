import numpy as np
from APO_env import ChineseCheckersEnv

env = ChineseCheckersEnv()

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Number of legal actions:", np.sum(info["action_mask"]))

for step in range(20):

    legal_actions = np.where(env.action_masks())[0]

    if len(legal_actions) == 0:
        print("No legal actions available.")
        break

    action = np.random.choice(legal_actions)

    print(f"\nStep {step}")
    print("Current player:", env.current_player)
    print("Chosen action:", action)

    obs, reward, terminated, truncated, info = env.step(action)

    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    if terminated or truncated:
        print("Episode finished.")
        print(info)
        break