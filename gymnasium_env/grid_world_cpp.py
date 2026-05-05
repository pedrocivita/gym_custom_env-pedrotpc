from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# Coverage Path Planning (CPP) environment based on GridWorld with obstacles.
#
# The agent must visit every free cell while avoiding obstacles.
#
# Observation space (v3):
#   - "agent": 7 floats [norm_x, norm_y, coverage_ratio,
#              unvisited_ratio_right, unvisited_ratio_up,
#              unvisited_ratio_left, unvisited_ratio_down]
#   - "global_map": 3 x size x size float32 tensor with channels:
#         channel 0 = obstacle_mask (1 where obstacle)
#         channel 1 = visited_mask  (1 where already visited)
#         channel 2 = agent_mask    (1 only at current agent position)
#
# Designed to be paired with MaskablePPO (sb3-contrib): the env exposes
# action_masks() so the policy never selects an action that would walk into
# a wall or obstacle.
#

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps

        self.visited = set()

        self._agent_location = np.array([-1, -1], dtype=int)

        # Cached structures (populated in reset())
        self._obstacle_set: set = set()
        self._obstacle_grid: np.ndarray = np.zeros((size, size), dtype=bool)

        # Pre-built coordinate arrays for vectorised _get_directional_signals()
        _r = np.arange(size)
        self._xs, self._ys = np.meshgrid(_r, _r, indexing='ij')

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.zeros(7, dtype=np.float32),
                high=np.ones(7, dtype=np.float32),
                dtype=np.float32,
            ),
            "global_map": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, size, size),
                dtype=np.float32,
            ),
        })

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    @property
    def total_free_cells(self):
        return self.size * self.size - len(self.obstacles_locations)

    @property
    def coverage_ratio(self):
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    def _visited_grid(self) -> np.ndarray:
        grid = np.zeros((self.size, self.size), dtype=bool)
        if self.visited:
            vis_arr = np.array(list(self.visited), dtype=int)
            grid[vis_arr[:, 0], vis_arr[:, 1]] = True
        return grid

    def _get_directional_signals(self, visited_grid: Optional[np.ndarray] = None):
        ax, ay = self._agent_location

        free_mask = ~self._obstacle_grid
        if visited_grid is None:
            visited_grid = self._visited_grid()
        unvisited_free = free_mask & ~visited_grid

        right_mask = self._xs > ax
        up_mask    = self._ys < ay
        left_mask  = self._xs < ax
        down_mask  = self._ys > ay

        signals = np.zeros(4, dtype=np.float32)
        for idx, dir_mask in enumerate((right_mask, up_mask, left_mask, down_mask)):
            region_free = free_mask & dir_mask
            total = region_free.sum()
            if total > 0:
                signals[idx] = float((unvisited_free & dir_mask).sum()) / float(total)
        return signals

    def _build_global_map(self, visited_grid: np.ndarray) -> np.ndarray:
        gmap = np.zeros((3, self.size, self.size), dtype=np.float32)
        gmap[0] = self._obstacle_grid.astype(np.float32)
        gmap[1] = visited_grid.astype(np.float32)
        gmap[2, self._agent_location[0], self._agent_location[1]] = 1.0
        return gmap

    def _get_obs(self):
        visited_grid = self._visited_grid()
        dir_signals = self._get_directional_signals(visited_grid)
        return {
            "agent": np.array([
                self._agent_location[0] / max(self.size - 1, 1),
                self._agent_location[1] / max(self.size - 1, 1),
                self.coverage_ratio,
                dir_signals[0],
                dir_signals[1],
                dir_signals[2],
                dir_signals[3],
            ], dtype=np.float32),
            "global_map": self._build_global_map(visited_grid),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def action_masks(self) -> np.ndarray:
        # True for legal actions (cell exists and is not an obstacle).
        # If the agent is fully boxed in, fall back to all-True so MaskablePPO
        # can still sample an action; the env will leave the agent in place.
        ax, ay = self._agent_location
        mask = np.zeros(4, dtype=bool)
        for action, direction in self._action_to_direction.items():
            nx, ny = ax + direction[0], ay + direction[1]
            if 0 <= nx < self.size and 0 <= ny < self.size and not self._obstacle_grid[nx, ny]:
                mask[action] = True
        if not mask.any():
            mask[:] = True
        return mask

    def _rebuild_obstacle_caches(self):
        self._obstacle_set = set(tuple(loc) for loc in self.obstacles_locations)
        self._obstacle_grid = np.zeros((self.size, self.size), dtype=bool)
        if self._obstacle_set:
            obs_arr = np.array(list(self._obstacle_set), dtype=int)
            self._obstacle_grid[obs_arr[:, 0], obs_arr[:, 1]] = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self.visited = set()

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        _tmp_obs_set: set = {tuple(self._agent_location)}
        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location.copy()
            while tuple(obstacle_location) in _tmp_obs_set:
                obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle_location)
            _tmp_obs_set.add(tuple(obstacle_location))

        self._rebuild_obstacle_caches()
        self.visited.add(tuple(self._agent_location))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if tuple(self._agent_location) in self._obstacle_set:
            self._agent_location = old_location

        self.count_steps += 1

        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self.visited

        # Reward shaping (v3): clean and bootstrap-friendly.
        # Action masking removes "stuck" cases, so no stuck penalty needed.
        # No truncation penalty — keeps value targets unbiased.
        reward = -0.05  # mild step cost
        if is_new_cell:
            reward += 1.0
            self.visited.add(current_pos)
        else:
            reward -= 0.25

        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage
        truncated = self.count_steps >= self.max_steps and not terminated

        if full_coverage:
            reward += 10.0 * (self.size / 5.0)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        for cell in self.visited:
            cell_arr = np.array(cell)
            pygame.draw.rect(
                canvas,
                (144, 238, 144),
                pygame.Rect(
                    pix_square_size * cell_arr,
                    (pix_square_size, pix_square_size),
                ),
            )

        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs,
                    (pix_square_size, pix_square_size),
                ),
            )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        font = pygame.font.SysFont(None, 24)
        coverage_text = font.render(
            f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
            True, (0, 0, 0)
        )
        canvas.blit(coverage_text, (5, 5))

        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x),
                             (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0),
                             (pix_square_size * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
