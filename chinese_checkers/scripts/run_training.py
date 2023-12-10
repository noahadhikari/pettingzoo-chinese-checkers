from chinese_checkers.env import chinese_checkers_env
from chinese_checkers.env.chinese_checkers_utils import action_to_move
from stable_baselines3 import A2C
import matplotlib.pyplot as plt

def train(board_size=1):
    print("in train")

    env = chinese_checkers_env.make_env(board_size)
    env.reset()
    
    max_iter = 100
    iter = 0
    for agent in env.agent_iter():
        if iter >= max_iter:
            break
        observation, reward, termination, truncation, info = env.last()

        # print(observation)
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
        if action:
            print(agent, action_to_move(action, board_size))
            env.step(action)
        iter += 1

        # render frame and wait til keypress to continue
        frame = env.render()
        plt.imshow(frame)
        plt.pause(0.05)
        plt.savefig("save.png")
        plt.draw()
    
    env.close()

def main():
    train()

if __name__ == "__main__":
    main()