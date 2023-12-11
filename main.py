import os
import pickle
import gymnasium
from gymnasium import spaces
import pygame
import numpy as np
import argparse
from policy_learning import qlearning, sarsa
import time
from pygame import image as pyg_image

import dweep_gym

ORDERING_MAP = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 6],
    [4, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 11, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--augment_rewards', '-a', action='store_true')
    parser.add_argument('--qlearning', '-q', action='store_true')
    parser.add_argument('--sarsa', '-s', action='store_true')
    parser.add_argument('--rec_pause', '-r', action='store_true')
    parser.add_argument('--load', '-l', action='store_true')

    args = parser.parse_args()

    game_map = ORDERING_MAP

    env_name = 'Dweep-v0'

    env = gymnasium.make(env_name, game_map=game_map, size=10, augment_rewards=args.augment_rewards)

    if args.qlearning:
        print("Learning with Q-Learning")
        Q, _, _, _ = qlearning(env, 
            episodes=20000, alpha=0.1, gamma=0.95, eps=0.2, verbose=True)
    elif args.sarsa:
        print("Learning with SARSA")
        Q, _, _, _ = sarsa(env, 
            episodes=120000, alpha=0.1, gamma=0.95, eps=0.2, verbose=True)
    elif args.load:
        print("Loading policy")
        Q = pickle.load(open('policy/q_policy.pkl', 'rb'))
    else:
        raise ValueError("Please specify what to do.")
    
    

    env = gymnasium.make(env_name, game_map=game_map, render_mode="human", size=10, augment_rewards=args.augment_rewards)
    env.reset()

    # print("Executing policy:")
    # for _ in range(30000):
    #     if not args.qlearning:
    #         action = env.action_space.sample()  # take a random action
    #     else:
    #         action = np.argmax(Q[env.get_state_idx()])

    # Visualize learned policy

    print("q shape: ", Q.shape)

    if not os.path.exists('policy'):
        os.makedirs('policy')
    
    pickle.dump(Q, open('policy/q_policy.pkl', 'wb'))
    
    done = False

    while not done:
        if args.rec_pause:
            time.sleep(10)
            args.rec_pause = False

        action = np.argmax(Q[env.get_state_idx()])
        print(action)
        next_state, reward, done, _, _ = env.step(action)
        
        print("Action: ", action)

    # Keep visualization for a few seconds for human watching
    time.sleep(10)

    env.close()