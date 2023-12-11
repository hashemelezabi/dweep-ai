from matplotlib.pylab import solve
import numpy as np
import random

def qlearning(env, alpha, gamma, eps, episodes=1000, return_on_success=False, verbose=False):
    num_states = env.num_states
    num_actions = env.action_space.n
    
    Q = np.zeros((num_states, num_actions))

    num_success = 0
    state = env.reset()
    avg_rewards = []

    solved = False
    first_solve_episode = None

    for i in range(episodes):
        if verbose:
            print(f"Episode {i+1}")
        state = env.reset()

        steps = 0
        total_rewards = 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < eps:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(Q[env.get_state_idx()]) # Exploit learned values

            next_state, reward, done, trunc, info = env.step(action) 
            total_rewards += reward

            # Q-Learning update
            best_next_action = np.argmax(Q[env.get_state_idx()])
            td_target = reward + gamma * Q[env.get_state_idx(), best_next_action]
            td_error = td_target - Q[env.get_state_idx(state), action]
            Q[env.get_state_idx(state), action] += alpha * td_error
            
            # new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            # q_table[state, action] = new_value

            if reward == env.TARGET_REWARD:
                num_success += 1
                if return_on_success:
                    return Q, i, avg_rewards, i
                if not solved:
                    solved = True
                    first_solve_episode = i


            state = next_state
            steps += 1

        avg_rewards.append(total_rewards / steps)
        
        if i % 10 == 0 and verbose:
            print(f"Avg return this episode: {total_rewards / steps}")
            print(f"{num_success} successes out of {i+1} episodes")

    return Q, i, avg_rewards, first_solve_episode

def sarsa(env, alpha, gamma, eps, episodes=1000, return_on_success=False, verbose=False):
    num_states = env.num_states
    num_actions = env.action_space.n
    
    Q = np.zeros((num_states, num_actions))

    num_success = 0
    state = env.reset()
    avg_rewards = []

    solved = False
    first_solve_episode = None

    for i in range(episodes):
        if verbose:
            print(f"Episode {i+1}")
        state = env.reset()

        steps = 0
        total_rewards = 0
        done = False

        last_exp_tuple = None
        
        while not done:
            state_idx = env.get_state_idx()
            if random.uniform(0, 1) < eps:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(Q[state_idx]) # Exploit learned values

            next_state, reward, done, trunc, info = env.step(action) 

            if last_exp_tuple is not None:
                last_state_idx, last_action, last_reward = last_exp_tuple
                Q[last_state_idx, last_action] += alpha * (last_reward + gamma * Q[state_idx, action] - Q[last_state_idx, last_action])

            last_exp_tuple = (state_idx, action, reward)

            total_rewards += reward    
            if reward == env.TARGET_REWARD:
                num_success += 1
                if return_on_success:
                    return Q, i, avg_rewards, i
                if not solved:
                    solved = True
                    first_solve_episode = i

            state = next_state
            steps += 1

        avg_rewards.append(total_rewards / steps)
        
        if i % 10 == 0 and verbose:
            print(f"Avg return this episode: {total_rewards / steps}")
            print(f"{num_success} successes out of {i+1} episodes")

    return Q, i, avg_rewards, first_solve_episode
