from new_dweep import HARD_MAP, DweepEnv
from policy_learning import qlearning, sarsa


if __name__ == '__main__':
    game_maps = [HARD_MAP]

    max_episodes = 200000

    for game_map in game_maps:
        print(f"Game map: {game_map}")

        env_reward = DweepEnv(game_map, size=10, augment_rewards=True)
        env_no_reward = DweepEnv(game_map, size=10, augment_rewards=False)

        Q, episodes = sarsa(env_reward, episodes=max_episodes, alpha=0.1, gamma=0.95, eps=0.2, return_on_success=True)

        print(f"Episodes to solve with reward augmentation using sarsa: {episodes}")

        Q, episodes = sarsa(env_no_reward, episodes=max_episodes, alpha=0.1, gamma=0.95, eps=0.2, return_on_success=True)

        print(f"Episodes to solve without reward augmentation using sarsa: {episodes}")

        print("")

        Q, episodes = qlearning(env_reward, episodes=max_episodes, alpha=0.1, gamma=0.95, eps=0.2, return_on_success=True)

        print(f"Episodes to solve with reward augmentation using qlearning: {episodes}")

        Q, episodes = qlearning(env_no_reward, episodes=max_episodes, alpha=0.1, gamma=0.95, eps=0.2, return_on_success=True)

        print(f"Episodes to solve without reward augmentation using qlearning: {episodes}")