from chinese_checkers.env import chinese_checkers_env
from chinese_checkers.env.chinese_checkers_utils import action_to_move
from stable_baselines3 import A2C
import matplotlib.pyplot as plt

def train(board_size=4):
    print("in train")

    env = chinese_checkers_env.make_env(board_size)
    env.reset()
    
    max_iter = 2
    iter = 0
    for agent in env.agent_iter():
        if iter >= max_iter:
            break
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # invalid action masking is optional and environment-dependent
            if "action_mask" in info:
                mask = info["action_mask"]
            elif isinstance(observation, dict) and "action_mask" in observation:
                mask = observation["action_mask"]
            else:
                mask = None
            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask)
        env.step(action)
        iter += 1

        # render frame and wait til keypress to continue
        frame = env.render()
        plt.imshow(frame)
        plt.pause(0.001)
        plt.savefig("train_result.png")
        plt.draw()
    
    env.close()

def main():
    train()

if __name__ == "__main__":
    main()