import argparse
import gym
from gym import spaces
import pygame
import numpy as np
import random

MAP = np.array([
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 4, 0, 0],
])

TARGET = 4
IMPASSABLE = [1, 2]

def get_target_distances(game_map):
    distances = np.full(game_map.shape, 1000)
    index_target = np.where(game_map == TARGET)

    distances[index_target] = 0

    while True:
        old_distances = distances.copy()
        for i in range(game_map.shape[0]):
            for j in range(game_map.shape[1]):
                square = game_map[i, j]
                
                if square in IMPASSABLE:
                    continue

                distances[i, j] = min(
                    old_distances[np.clip(i + 1, 0, game_map.shape[0] - 1), j] + 1,
                    old_distances[np.clip(i - 1, 0, game_map.shape[0] - 1), j] + 1,
                    old_distances[i, np.clip(j + 1, 0, game_map.shape[1] - 1)] + 1,
                    old_distances[i, np.clip(j - 1, 0, game_map.shape[1] - 1)] + 1,
                    old_distances[i, j],
                )

        if np.all(old_distances == distances):
            break

    return distances


class DweepEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, augment=False):
        self.size = size  # The size of the square grid
        self.window_size = 520  # The size of the PyGame window

        self.augment = augment
        if augment:
            self.distances = get_target_distances(MAP)


        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(100), # 100 possible Dweep locations
                "target": spaces.Discrete(2), # 2 possible target locations for now
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]), # right
            1: np.array([0, -1]), # up
            2: np.array([-1, 0]), # left
            3: np.array([0, 1]), # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def get_state_idx(self, state=None):
        if state is None:
            obs = self._get_obs()
        else:
            obs = state[0] if isinstance(state, tuple) else state

        x, y = obs['agent'][0], obs['agent'][1]
        i = x * self.size + y
        if np.array_equal(obs['target'], np.array([0, 7])):
            i += 100 # If it's the other possible target location
        return i

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = np.array([0, 0]) # Top-left
        self._target_location = np.array([9, 7]) # In bottom-right corner

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        new_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        x, y = new_location

        if MAP[x, y] == 3:
            # This is a laser, so we die
            self._agent_location = new_location
            reward = -10
            terminated = True
        elif MAP[x, y] in [1, 2]:
            # This is a wall or laser generator, so we stay at old location
            reward = -2
            terminated = False
        elif MAP[x, y] == 0:
            # Dweep moves to a new square
            self._agent_location = new_location
            reward = -1 # Give -1 reward to encourage Dweep to move to destination
            terminated = False
        elif MAP[x, y] == 4:
            self._agent_location = new_location
            reward = 10 # Binary sparse rewards
            terminated = True
        else:
            raise ValueError("Invalid entry in map")

        if self.augment:
            reward += 0.01 * -self.distances[new_location[0], new_location[1]]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {}
        # return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (50, 255, 255),
            pygame.Rect(
                pix_square_size * np.flip(self._target_location),
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent Dweep
        pygame.draw.circle(
            canvas,
            (255, 0, 255),
            (np.flip(self._agent_location) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw the walls, laser, and laser generator
        # TODO: Remove this hardcoded size
        for i in range(10):
            for j in range(10):
                loc = np.array([i, j])
                if MAP[i, j] == 1: # Draw wall
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.flip(loc),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif MAP[i, j] == 2: # Draw laser generator
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 255),
                        pygame.Rect(
                            pix_square_size * np.flip(loc),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif MAP[i, j] == 3: # Draw laser
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            pix_square_size * np.flip(loc + 0.25),
                            (pix_square_size / 2, pix_square_size),
                        ),
                    )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

def main(augment_rewards):
    # env = DweepEnv(render_mode="human", size=10)
    env = DweepEnv(size=10, augment=augment_rewards)

    num_states = 200
    num_actions = 4
    
    Q = np.zeros((num_states, num_actions))
    alpha = 0.1
    gamma = 0.95
    eps = 0.1

    episodes = 200
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

    env = DweepEnv(render_mode="human", size=10)
    env.reset()
    for _ in range(1000):
        action = np.argmax(Q[env.get_state_idx()])
        observation, reward, _, _, _ = env.step(action)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--render', action='store_true')
    parser.add_argument('--augment_rewards', '-a', action='store_true')
    # parser.add_argument('--qlearning', '-q', action='store_true')
    args = parser.parse_args()

    # env = DweepEnv(render_mode="human", size=10)
    # env.reset()
    # for _ in range(1000):
    #     action = env.action_space.sample()  # take a random action
    #     env.step(action)
    # env.close()
    main(args.augment_rewards)