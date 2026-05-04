from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# Coverage Path Planning (CPP) environment based on GridWorld with obstacles.
#
# The agent must visit as many free cells as possible while avoiding obstacles.
#
# Observation space:
#   - "agent": 7 floats [norm_x, norm_y, coverage_ratio,
#              unvisited_ratio_right, unvisited_ratio_up,
#              unvisited_ratio_left, unvisited_ratio_down]
#   - "neighbors": 5x5 matrix centered on agent
#     (0=free/unvisited, 1=obstacle/wall, 2=visited)
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
        self.consecutive_revisits = 0

        self.visited = set()

        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.zeros((5, 5), dtype=int)

        # --- Cached structures (populated in reset()) ---
        # Set of (x, y) tuples for O(1) obstacle lookup in step()
        self._obstacle_set: set = set()
        # Boolean grid [size x size]: True where obstacle exists
        # Used by set_neighbors() for fast slicing instead of per-cell any()
        self._obstacle_grid: np.ndarray = np.zeros((size, size), dtype=bool)

        # Pre-built coordinate arrays for vectorized _get_directional_signals()
        # xs[i,j] = i, ys[i,j] = j  (shape: size x size)
        _r = np.arange(size)
        self._xs, self._ys = np.meshgrid(_r, _r, indexing='ij')  # shape (size, size)

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.zeros(7, dtype=np.float32),
                high=np.ones(7, dtype=np.float32),
                dtype=np.float32
            ),
            "neighbors": gym.spaces.Box(
                low=np.zeros((5, 5), dtype=np.float32),
                high=np.full((5, 5), 2.0, dtype=np.float32),
                dtype=np.float32
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

    def _get_directional_signals(self):
        ax, ay = self._agent_location

        # free_mask: True for cells that are NOT obstacles — shape (size, size)
        free_mask = ~self._obstacle_grid  # pre-built boolean grid

        # Build a visited boolean grid on-the-fly from the visited set.
        # For typical grid sizes (<=20) this is faster than maintaining a
        # synchronized array because visited grows incrementally and the
        # set iteration is cheap in Python relative to numpy overhead at
        # small sizes; for larger grids the vectorised ops dominate anyway.
        visited_grid = np.zeros((self.size, self.size), dtype=bool)
        if self.visited:
            vis_arr = np.array(list(self.visited), dtype=int)  # (n, 2)
            visited_grid[vis_arr[:, 0], vis_arr[:, 1]] = True

        # unvisited_free: cells that are free AND not yet visited
        unvisited_free = free_mask & ~visited_grid

        # Directional masks using the pre-built coordinate grids
        # self._xs[i,j] == i (x-coordinate), self._ys[i,j] == j (y-coordinate)
        right_mask = self._xs > ax   # x > ax
        up_mask    = self._ys < ay   # y < ay  (up = decreasing y)
        left_mask  = self._xs < ax   # x < ax
        down_mask  = self._ys > ay   # y > ay

        signals = np.zeros(4, dtype=np.float32)
        for idx, dir_mask in enumerate((right_mask, up_mask, left_mask, down_mask)):
            region_free = free_mask & dir_mask
            total = region_free.sum()
            if total > 0:
                signals[idx] = float((unvisited_free & dir_mask).sum()) / float(total)
            # else stays 0.0

        return signals

    def _get_obs(self):
        dir_signals = self._get_directional_signals()
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
            "neighbors": self._neighbors.astype(np.float32),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def set_neighbors(self, obstacles_locations):
        # Build the 5x5 neighbors matrix using array slicing on a padded grid
        # rather than a Python double-loop + per-cell any(np.array_equal(...)).
        #
        # Grid encoding: 0 = free/unvisited, 1 = obstacle/wall, 2 = visited
        # Padding with 1s represents out-of-bounds (wall).

        ax, ay = self._agent_location

        # Padded grid: (size+4) x (size+4), pre-filled with 1 (wall/OOB).
        # Interior region [2 : size+2, 2 : size+2] corresponds to the real grid.
        pad = 2
        padded = np.ones((self.size + 2 * pad, self.size + 2 * pad), dtype=int)

        # Mark free interior cells as 0
        padded[pad:pad + self.size, pad:pad + self.size] = 0

        # Mark obstacles as 1 (already 1 from padding, but set explicitly for interior)
        if self._obstacle_set:
            obs_arr = np.array(list(self._obstacle_set), dtype=int)  # (n_obs, 2)
            padded[obs_arr[:, 0] + pad, obs_arr[:, 1] + pad] = 1

        # Mark visited cells as 2
        if self.visited:
            vis_arr = np.array(list(self.visited), dtype=int)  # (n_vis, 2)
            # Only mark if not an obstacle (obstacles stay as 1)
            vis_x = vis_arr[:, 0] + pad
            vis_y = vis_arr[:, 1] + pad
            not_obs = padded[vis_x, vis_y] != 1
            padded[vis_x[not_obs], vis_y[not_obs]] = 2

        # Slice the 5x5 region centered on the agent.
        # Agent is at (ax, ay) in the real grid → (ax + pad, ay + pad) in padded.
        cx, cy = ax + pad, ay + pad
        # matrix[i][j] where i is row (y-offset) and j is col (x-offset),
        # matching the original loop: nx = ax + (j-2), ny = ay + (i-2)
        self._neighbors = padded[cx - 2:cx + 3, cy - 2:cy + 3].T
        # Note: padded is indexed [x, y]; the original matrix[i][j] = matrix[row, col]
        # maps row→y-offset, col→x-offset, so we transpose to get [y_offset, x_offset].

    def _rebuild_obstacle_caches(self):
        """Rebuild _obstacle_set and _obstacle_grid from self.obstacles_locations.

        Called once per reset() after the obstacle list is finalised.
        """
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
        self.consecutive_revisits = 0

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Build obstacle set incrementally during placement (O(1) lookup).
        _tmp_obs_set: set = {tuple(self._agent_location)}

        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location.copy()
            while tuple(obstacle_location) in _tmp_obs_set:
                obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle_location)
            _tmp_obs_set.add(tuple(obstacle_location))

        # Commit obstacle caches (set + bool grid) once, for use throughout the episode.
        self._rebuild_obstacle_caches()

        self.visited.add(tuple(self._agent_location))

        self.set_neighbors(self.obstacles_locations)

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

        # O(1) obstacle check via cached set instead of O(n) any(np.array_equal(...))
        if tuple(self._agent_location) in self._obstacle_set:
            self._agent_location = old_location

        self.set_neighbors(self.obstacles_locations)
        self.count_steps += 1

        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)

        reward = -0.1

        if stayed_in_place:
            reward -= 0.5
            self.consecutive_revisits += 1
        elif is_new_cell:
            reward += 1.0
            self.visited.add(current_pos)
            self.consecutive_revisits = 0
        else:
            penalty = 0.3 * (1.0 + self.consecutive_revisits * 0.1)
            reward -= min(penalty, 1.0)
            self.consecutive_revisits += 1

        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage

        if full_coverage:
            reward += 10.0 * (self.size / 5.0)

        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            reward -= 5.0
        else:
            truncated = False

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
