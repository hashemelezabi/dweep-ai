import gym
from gym import spaces
import pygame
import numpy as np
import argparse
from policy_learning import qlearning, sarsa
import time

"""
0: Empty Square
1: Wall
2: Up Laser Generator
3: Down Laser Generator
4: Right Laser Generator
5: Left Laser Generator
6: Target
7: Bomb
8: Freeze Plate
9: Right Mirror
10: Left Mirror
11: Water Bucket
"""

GO_ABOVE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 10, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 6, 0, 0],
])

GO_BELOW = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 9, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 6, 0, 0],
])

NEED_DIAGONAL = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 9, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 6, 0, 0],
])

HARD_MAP = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [11, 2, 0, 0, 0, 2, 0, 0, 6, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

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

####################################
####### UTILS #########
####################################

class Tiles:
    EMPTY = 0
    WALL = 1
    ULASER = 2
    DLASER = 3
    RLASER = 4
    LLASER = 5
    TARGET = 6
    BOMB = 7
    FREEZE = 8
    LMIRROR = 9
    RMIRROR = 10
    BUCKET = 11

    IMPASSABLE = [WALL, ULASER, DLASER, RLASER, LLASER, LMIRROR, RMIRROR]
    PASSABLE = [EMPTY, TARGET, BOMB, FREEZE, BUCKET]


def get_triangle_points(loc, pix_square_size, laser_gen_dir):
    """
    2: Up Laser Generator
    3: Down Laser Generator
    4: Right Laser Generator
    5: Left Laser Generator
    """
    if laser_gen_dir == 2:
        p1 = ((loc[0] + 0.5) * pix_square_size, loc[1] * pix_square_size)
        p2 = (loc[0] * pix_square_size, (loc[1] + 1) * pix_square_size)
        p3 = ((loc[0] + 1) * pix_square_size, (loc[1] + 1) * pix_square_size)
    elif laser_gen_dir == 3:
        p1 = ((loc[0] + 0.5) * pix_square_size, (loc[1] + 1) * pix_square_size)
        p2 = tuple(loc * pix_square_size)
        p3 = ((loc[0] + 1) * pix_square_size, loc[1] * pix_square_size)
    elif laser_gen_dir == 4:
        p1 = ((loc[0] + 1) * pix_square_size, (loc[1] + 0.5) * pix_square_size)
        p2 = tuple(loc * pix_square_size)
        p3 = (loc[0] * pix_square_size, (loc[1] + 1) * pix_square_size)
    elif laser_gen_dir == 5:
        p1 = (loc[0] * pix_square_size, (loc[1] + 0.5) * pix_square_size)
        p2 = ((loc[0] + 1) * pix_square_size, loc[1] * pix_square_size)
        p3 = ((loc[0] + 1) * pix_square_size, (loc[1] + 1) * pix_square_size)
    else:
        raise ValueError

    return [p1, p2, p3]

def get_mirror_points(loc, d, mirror):
    """
    loc: [x, y] of square in Pygame grid
    d: length of one side of square
    mirror: whether right (9) or left (10) mirror
    """
    if mirror == 9: # Right mirror
        p1 = ((loc[0] + 0.9) * d, loc[1] * d)
        p2 = ((loc[0] + 1) * d, (loc[1] + 0.1) * d)
        p3 = ((loc[0] + 0.1) * d, (loc[1] + 1) * d)
        p4 = (loc[0] * d, (loc[1] + 0.9) * d)
    elif mirror == 10:
        p1 = ((loc[0] + 0.1) * d, loc[1] * d)
        p2 = (loc[0] * d, (loc[1] + 0.1) * d)
        p3 = ((loc[0] + 0.9) * d, (loc[1] + 1) * d)
        p4 = ((loc[0] + 1) * d, (loc[1] + 0.9) * d)
    else:
        raise ValueError
    return [p1, p2, p3, p4]

def get_target_distances(game_map):
    distances = np.full(game_map.shape, 1000)
    index_target = np.where(game_map == Tiles.TARGET)

    distances[index_target] = 0

    while True:
        old_distances = distances.copy()
        for i in range(game_map.shape[0]):
            for j in range(game_map.shape[1]):
                square = game_map[i, j]
                
                if square in Tiles.IMPASSABLE:
                    continue

                distances[i, j] = min(
                    old_distances[np.clip(i + 1, 0, game_map.shape[0] - 1), j] + 1,
                    old_distances[np.clip(i - 1, 0, game_map.shape[0] - 1), j] + 1,
                    old_distances[i, np.clip(j + 1, 0, game_map.shape[1] - 1)] + 1,
                    old_distances[i, np.clip(j - 1, 0, game_map.shape[1] - 1)] + 1,
                    old_distances[np.clip(i + 1, 0, game_map.shape[0] - 1), np.clip(j + 1, 0, game_map.shape[1] - 1)] + 1,
                    old_distances[np.clip(i - 1, 0, game_map.shape[0] - 1), np.clip(j - 1, 0, game_map.shape[1] - 1)] + 1,
                    old_distances[np.clip(i + 1, 0, game_map.shape[0] - 1), np.clip(j - 1, 0, game_map.shape[1] - 1)] + 1,
                    old_distances[np.clip(i - 1, 0, game_map.shape[0] - 1), np.clip(j + 1, 0, game_map.shape[1] - 1)] + 1,
                    old_distances[i, j],
                )

        # print(distances)
        if np.all(old_distances == distances):
            break

    return distances

####################################
####### END UTILS #########
####################################

class DweepEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def find_target(self, game_map):
        for i in range(game_map.shape[0]):
            for j in range(game_map.shape[1]):
                if game_map[i, j] == 6:
                    return np.array([i, j])

    def __init__(self, game_map, render_mode=None, size=5, augment_rewards=False):
        self.size = size  # The size of the square grid
        self.window_size = 520  # The size of the PyGame window

        # handle reward augmentation
        self.augment_rewards = augment_rewards
        self.distances = get_target_distances(game_map)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "cursor": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "mirror_placed": spaces.Discrete(3),
                "wet": spaces.Discrete(2),
                "has_bucket": spaces.Discrete(3),
            }
        )

        self.num_states = size * size * size * size * 3 * 2 * 3

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        # We have 4 more actions corresponding to diagonal movement (8 total)
        self.action_space = spaces.Discrete(8 + 4 + 2 + 1)

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
            4: np.array([1, -1]), # up and right
            5: np.array([-1, -1]), # up and left
            6: np.array([1, 1]), # down and right
            7: np.array([-1, 1]), # up and left
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

        self.initial_target_location = self.find_target(game_map)
        self.base_map = game_map.copy()
        self.game_map = game_map.copy() # This map can change
        self.laser_map = np.zeros(self.game_map.shape) # Keeps track of laser paths
        self.laser_is_vertical = np.zeros(self.game_map.shape) # Used for visualizing the laser
        self.update_lasers()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = np.array([0, 0]) # Top-left
        self._cursor_location = np.array([0, 0]) # Top-left
        self._target_location = self.initial_target_location
        self._mirror_placed = 0
        self._wet = 0
        self._has_bucket = 0

        self.game_map = self.base_map.copy()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        reward = 0
        terminated = False

        if action == 14:
            # water bucket case
            if self._has_bucket == 1 and self._wet == 0:
                # print("used water bucket")
                self._wet = 1
                self._has_bucket = 2
                if self.augment_rewards:
                    reward = 0.1
            else:
                if self.augment_rewards:
                    reward = -0.1
        elif action >= 12:
            # mirror placement case
            if self._mirror_placed == 0 and not np.array_equal(self._cursor_location, self._agent_location) and not np.array_equal(self._cursor_location, self._target_location):
                # print("laser map: ", self.laser_map[self._cursor_location[0], self._cursor_location[1]])
                if self.game_map[self._cursor_location[0], self._cursor_location[1]] == Tiles.EMPTY and self.laser_map[self._cursor_location[0], self._cursor_location[1]] == True:
                    self._mirror_placed = action - 12 + 1
                    if action == 12:
                        self.game_map[self._cursor_location[0], self._cursor_location[1]] = Tiles.LMIRROR
                    else:
                        self.game_map[self._cursor_location[0], self._cursor_location[1]] = Tiles.RMIRROR

                    self.update_lasers()

                    if self.augment_rewards:
                        # print("good mirror placement: ", self._cursor_location)
                        reward += 0.2
            else:
                # print("bad mirror placement")
                if self.augment_rewards:
                    reward = -0.1

        elif action >= 8:
            # cursor move case
            if self._mirror_placed == 0:
                direction = self._action_to_direction[action - 8]
                new_cursor_location = np.clip(
                    self._cursor_location + direction, 0, self.size - 1
                )

                # print("cursor move", self._cursor_location, new_cursor_location)
                if self.augment_rewards and (np.array_equal(new_cursor_location, self._agent_location) or np.array_equal(new_cursor_location, self._cursor_location)):
                    # print("bad cursor move")
                    reward = -0.01

                self._cursor_location = new_cursor_location
            else:
                # can't move cursor, as this represents a mirror placement
                if self.augment_rewards:
                    reward = -0.01
        else:
            # Map the action (element of {0,1,2,3,4,5,6,7}) to the direction we walk in
            direction = self._action_to_direction[action]
            # We use `np.clip` to make sure we don't leave the grid
            new_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

            x, y = new_location

            # Check for laser beam collision
            if self.game_map[x, y] in Tiles.IMPASSABLE:
                # This is a wall or laser generator or mirror, so we stay at old location
                reward = 0
                terminated = False
            elif self.laser_map[x, y]:
                # This is a laser, so we die
                if not self._wet:
                    self._agent_location = new_location
                    reward = -1
                    # print("laser death")
                    terminated = True
                else:
                    self._agent_location = new_location
                    reward = 0
                    terminated = False
                    self._wet = 0
            elif self.game_map[x, y] in [0, 8, Tiles.BUCKET]:
                # Dweep moves to a new square
                self._agent_location = new_location
                
                if self.game_map[x, y] == Tiles.BUCKET:
                    # print("picked up water bucket")
                    self._has_bucket = 1
                    self.game_map[x, y] = 0
                    if self.augment_rewards:
                        reward = 0.1

                reward = 0
                terminated = False
            elif self.game_map[x, y] == Tiles.TARGET:
                self._agent_location = new_location
                reward = 1000 # Binary sparse rewards
                terminated = True
            else:
                raise ValueError("Invalid entry in map")

            if self.augment_rewards:
                reward += 0.01 * -self.distances[new_location[0], new_location[1]]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location, "cursor": self._cursor_location, 
                "mirror_placed": self._mirror_placed, "wet": self._wet, "has_bucket": self._has_bucket}
    
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

        # Now we draw the cursor
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (np.flip(self._cursor_location) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw the walls, laser, laser generator, and freeze plates
        for i in range(self.size):
            for j in range(self.size):
                loc = np.array([j, i]) # Flip it to match coordinate system of PyGame
                if self.game_map[i, j] == 1: # Draw wall
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            pix_square_size * loc,
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.game_map[i, j] in [2, 3, 4, 5]: # Draw laser generator
                    pygame.draw.polygon(
                        canvas,
                        (0, 0, 255),
                        get_triangle_points(loc, pix_square_size, self.game_map[i, j])
                    )
                elif self.game_map[i, j] == 8: # Draw freeze plate
                    pygame.draw.rect(
                        canvas,
                        (0, 100, 100),
                        pygame.Rect(
                            pix_square_size * loc,
                            (pix_square_size, pix_square_size),
                        )
                    )
                elif self.game_map[i, j] == Tiles.BUCKET and self._has_bucket == 0: # Draw water bucket
                    pygame.draw.rect(
                        canvas,
                        (50, 50, 255),
                        pygame.Rect(
                            pix_square_size * loc,
                            (pix_square_size, pix_square_size),
                        )
                    )

                # Draw laser beam
                if self.laser_map[i, j]:
                    if self.laser_is_vertical[i, j]:
                        topleft = ((loc[0] + 0.4) * pix_square_size, loc[1] * pix_square_size)
                        dim = (pix_square_size * 0.2, pix_square_size)
                    else:
                        topleft = (loc[0] * pix_square_size, (loc[1] + 0.4) * pix_square_size)
                        dim = (pix_square_size, pix_square_size * 0.2)
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            topleft,
                            dim,
                        )
                    )

                # Draw mirror
                if self.game_map[i, j] in [Tiles.LMIRROR, Tiles.RMIRROR]:
                    silver_color = (192, 192, 192)
                    pygame.draw.polygon(
                        canvas,
                        silver_color,
                        get_mirror_points(loc, pix_square_size, self.game_map[i, j]),
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

    def get_state_idx(self, state=None):
        if state is None:
            obs = self._get_obs()
        else:
            obs = state[0] if isinstance(state, tuple) else state

        x, y = obs['agent'][0], obs['agent'][1]
        cursor_x, cursor_y = obs['cursor'][0], obs['cursor'][1]

        mirror_placed = obs['mirror_placed']
        is_wet = obs['wet']
        has_bucket = obs['has_bucket']

        return np.ravel_multi_index((x, y, cursor_x, cursor_y, mirror_placed, is_wet, has_bucket), (self.size, self.size, self.size, self.size, 3, 2, 3))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # Helper method to update laser paths
    def update_lasers(self):
        self.laser_map = np.zeros(self.game_map.shape) # Keeps track of laser paths
        self.laser_is_vertical = np.zeros(self.game_map.shape) # Used for visualizing the laser

        direction = np.array([0, 0])
        for row in range(self.game_map.shape[0]):
            for col in range(self.game_map.shape[1]):
                is_vertical = 0
                if self.game_map[row][col] == 2:
                    direction = np.array([-1, 0])
                    is_vertical = 1
                elif self.game_map[row][col] == 3:
                    direction = np.array([1, 0])
                    is_vertical = 1
                elif self.game_map[row][col] == 4:
                    direction = np.array([0, 1])
                elif self.game_map[row][col] == 5:
                    direction = np.array([0, -1])
                else:
                    continue
                # Following code only executes if the object is a laser generator
                # Trace the laser's path until it hits a wall, changing direction at mirrors
                loc = np.array([row, col])
                loc += direction
                while (loc[0] < self.laser_map.shape[0] and loc[0] >= 0 and loc[1] < self.laser_map.shape[1] and loc[1] >= 0):
                    if self.game_map[loc[0]][loc[1]] == 1:
                        break
                    elif self.game_map[loc[0]][loc[1]] in [2, 3, 4, 5]:
                        self.laser_map[loc[0]][loc[1]] = 1
                        self.laser_is_vertical[loc[0]][loc[1]] = is_vertical
                        break
                    elif self.game_map[loc[0]][loc[1]] == 9: # Right mirror
                        self.laser_map[loc[0]][loc[1]] = 1
                        self.laser_is_vertical[loc[0]][loc[1]] = is_vertical
                        direction = np.array([direction[1] * -1, direction[0] * -1])
                        is_vertical = 0 if is_vertical else 1 # Invert is_vertical, since we hit a mirror
                    elif self.game_map[loc[0]][loc[1]] == 10:
                        self.laser_map[loc[0]][loc[1]] = 1
                        self.laser_is_vertical[loc[0]][loc[1]] = is_vertical
                        direction = np.array([direction[1], direction[0]])
                        is_vertical = 0 if is_vertical else 1 # Invert is_vertical, since we hit a mirror
                    else:
                        self.laser_map[loc[0]][loc[1]] = 1
                        self.laser_is_vertical[loc[0]][loc[1]] = is_vertical
                    loc += direction
                
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--augment_rewards', '-a', action='store_true')
    parser.add_argument('--qlearning', '-q', action='store_true')
    parser.add_argument('--sarsa', '-s', action='store_true')
    args = parser.parse_args()

    game_map = ORDERING_MAP

    if args.qlearning:
        print("Learning with Q-Learning")
        Q = qlearning(DweepEnv(game_map, size=10, augment_rewards=args.augment_rewards), 
        episodes=10000, alpha=0.1, gamma=0.95, eps=0.2)
    elif args.sarsa:
        print("Learning with SARSA")
        Q = sarsa(DweepEnv(game_map, size=10, augment_rewards=args.augment_rewards), 
        episodes=120000, alpha=0.1, gamma=0.95, eps=0.2)

    env = DweepEnv(game_map, render_mode="human", size=10, augment_rewards=args.augment_rewards)
    env.reset()

    print("Executing policy:")
    for _ in range(30000):
        if not args.qlearning:
            action = env.action_space.sample()  # take a random action
        else:
            action = np.argmax(Q[env.get_state_idx()])

    # Visualize learned policy
    done = False
    while not done:
        action = np.argmax(Q[env.get_state_idx()])
        next_state, reward, done, _, _ = env.step(action)

    # Keep visualization for a few seconds for human watching
    time.sleep(10)

    env.close()