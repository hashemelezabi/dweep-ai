import os
from new_dweep import GO_ABOVE, HARD_MAP, ORDERING_MAP, DweepEnv
from policy_learning import qlearning, sarsa
import matplotlib.pyplot as plt

def plot_avg_rewards(avg_rewards, game_map, method, augmented_rewards, first_solve_episode=None):
    stride = 100

    avg_rewards = [avg_rewards[i:i+stride] for i in range(0, len(avg_rewards), stride)]
    avg_rewards = [sum(batch) / len(batch) for batch in avg_rewards]

    episodes = [i * stride for i in range(len(avg_rewards))]

    plt.plot(episodes, avg_rewards)

    if first_solve_episode is not None:
        plt.axvline(x=first_solve_episode, color='r', linestyle='--')

    plt.xlabel("Episode")
    plt.ylabel("Average Reward (smoothed)")
    plt.title(f"Average Reward for {game_map} with {'Augmented' if augmented_rewards else 'Regular'} Rewards")
    plt.savefig(f"plots/{game_map}_{method}_{'aug' if augmented_rewards else 'reg'}.png")
    plt.close()

if __name__ == '__main__':
    # game_maps = {"Hard Map": HARD_MAP, "Ordering Map": ORDERING_MAP}
    # game_maps = {"Multi-Laser Map": ORDERING_MAP}
    game_maps = {"Simple Map": GO_ABOVE}

    max_episodes = 80000

    if not os.path.exists("plots"):
        os.makedirs("plots")

    for name, game_map in game_maps.items():
        print(f"Game map: {game_map}")

        env_reward = DweepEnv(game_map, size=10, augment_rewards=True)
        env_no_reward = DweepEnv(game_map, size=10, augment_rewards=False)

        # Q, episodes, avg_rewards, first_solve = sarsa(env_reward, episodes=max_episodes, alpha=0.1, gamma=0.95, eps=0.2, verbose=True)
        # plot_avg_rewards(avg_rewards, name, "sarsa", True, first_solve_episode=first_solve)

        # print(f"Episodes to solve with reward augmentation using sarsa: {first_solve}")

        # Q, episodes, avg_rewards, first_solve = sarsa(env_no_reward, episodes=max_episodes, alpha=0.1, gamma=0.95, eps=0.2, verbose=True)
        # plot_avg_rewards(avg_rewards, name, "sarsa", False, first_solve_episode=first_solve)

        # print(f"Episodes to solve without reward augmentation using sarsa: {first_solve}")

        # print("")

        Q, episodes, avg_rewards, first_solve = qlearning(env_reward, episodes=max_episodes, alpha=0.1, gamma=0.95, eps=0.2, verbose=True)
        plot_avg_rewards(avg_rewards, name, "qlearning", True, first_solve_episode=first_solve)

        print(f"Episodes to solve with reward augmentation using qlearning: {first_solve}")

        Q, episodes, avg_rewards, first_solve = qlearning(env_no_reward, episodes=max_episodes, alpha=0.1, gamma=0.95, eps=0.2, verbose=True)
        plot_avg_rewards(avg_rewards, name, "qlearning", False, first_solve_episode=first_solve)

        print(f"Episodes to solve without reward augmentation using qlearning: {first_solve}")