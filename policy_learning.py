import numpy as np
import random

def qlearning(env, alpha, gamma, eps, episodes=1000):
    num_states = env.num_states
    num_actions = env.action_space.n
    
    Q = np.zeros((num_states, num_actions))

    num_success = 0
    state = env.reset()

    for i in range(episodes):
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

            if reward == 10:
                num_success += 1

            state = next_state
            steps += 1
        
        if i % 10 == 0:
            print(f"Avg return this episode: {total_rewards / steps}")
            print(f"{num_success} successes out of {i+1} episodes")

    return Q
