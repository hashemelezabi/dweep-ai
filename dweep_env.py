import gym
from gym import spaces
import pygame
import numpy as np

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


class DweepEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 520  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
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
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
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

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = np.array([0, 9]) # Top-left
        self._target_location = np.array([8, 0]) # In bottom-right corner

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
            reward = -1
            terminated = True
        elif MAP[x, y] in [1, 2]:
            # This is a well or laser generator, so we stay at old location
            reward = 0
            terminated = False
        elif MAP[x, y] == 0:
            # Dweep moves to a new square
            self._agent_location = new_location
            reward = 0
            terminated = False
        elif MAP[x, y] == 4:
            self._agent_location = new_location
            reward = 1 # Binary sparse rewards
            terminated = True
        else:
            raise ValueError("Invalid entry in map")

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
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent Dweep
        pygame.draw.circle(
            canvas,
            (255, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
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
                            pix_square_size * loc,
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif MAP[i, j] == 2: # Draw laser generator
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 255),
                        pygame.Rect(
                            pix_square_size * loc,
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif MAP[i, j] == 3: # Draw laser
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            pix_square_size * loc,
                            (pix_square_size / 4, pix_square_size),
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


if __name__ == '__main__':
    env = DweepEnv(render_mode="human", size=10)
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()  # take a random action
        env.step(action)
    env.close()