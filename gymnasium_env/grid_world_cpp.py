from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# Coverage Path Planning (CPP) environment — partial-observability compliant,
# 5x5-windowed (v3.2 / curriculum-friendly).
#
# Observation:
#   - "agent": 7 floats — pose, coverage, 4 directional unvisited ratios
#       computed only over what the agent already visited.
#   - "neighbors": 5x5 sensor view centred on the agent
#       (0 = free, 1 = obstacle/wall, 2 = visited).
#   - "visited_neighbors": 5x5 binary memory window — 1 where the agent has
#       already stepped (inside the same 5x5 window), 0 otherwise.
#
# Both spatial inputs are size-invariant (always 5x5), so weights transfer
# between grid sizes (5x5 -> 10x10 -> 20x20). This is the key change vs v3.1:
# the previous global visited_map had shape (size, size), which broke the
# Linear layer between sizes and prevented curriculum learning.
#
# Action masking (used by MaskablePPO) only masks moves that would leave the
# grid. Obstacles are NOT masked: the agent must discover them through the
# 5x5 sensor and the stuck penalty.
#
# Reward (v3.2):
#   step base               -0.05
#   new cell                +1.0
#   revisit                 -0.25
#   stuck (vs obstacle)     -0.5
#   25% / 50% / 75% milestone  +2.0 each (one-time per episode)
#   full coverage           +10 * (size / 5)
#   truncation              0
#

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    WINDOW = 5  # spatial window radius parameter (window is 2*PAD+1 = 5 wide)
    PAD = 2

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps
        self.consecutive_revisits = 0

        # Coverage milestones already awarded in this episode.
        self._milestones_awarded: set = set()

        self.visited = set()

        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.zeros((self.WINDOW, self.WINDOW), dtype=int)
        self._visited_neighbors = np.zeros((self.WINDOW, self.WINDOW), dtype=int)

        # Cached structures (NOT exposed to the agent).
        self._obstacle_set: set = set()
        self._obstacle_grid: np.ndarray = np.zeros((size, size), dtype=bool)

        # Pre-built coordinate arrays for vectorised _get_directional_signals().
        _r = np.arange(size)
        self._xs, self._ys = np.meshgrid(_r, _r, indexing='ij')

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.zeros(7, dtype=np.float32),
                high=np.ones(7, dtype=np.float32),
                dtype=np.float32,
            ),
            "neighbors": gym.spaces.Box(
                low=np.zeros((self.WINDOW, self.WINDOW), dtype=np.float32),
                high=np.full((self.WINDOW, self.WINDOW), 2.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "visited_neighbors": gym.spaces.Box(
                low=np.zeros((self.WINDOW, self.WINDOW), dtype=np.float32),
                high=np.ones((self.WINDOW, self.WINDOW), dtype=np.float32),
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
        if visited_grid is None:
            visited_grid = self._visited_grid()
        unvisited_known = ~visited_grid

        right_mask = self._xs > ax
        up_mask    = self._ys < ay
        left_mask  = self._xs < ax
        down_mask  = self._ys > ay

        signals = np.zeros(4, dtype=np.float32)
        for idx, dir_mask in enumerate((right_mask, up_mask, left_mask, down_mask)):
            total = dir_mask.sum()
            if total > 0:
                signals[idx] = float((unvisited_known & dir_mask).sum()) / float(total)
        return signals

    def _build_windows(self, visited_grid: np.ndarray):
        """Compute neighbors (5x5) and visited_neighbors (5x5) via padded slicing."""
        ax, ay = self._agent_location
        pad = self.PAD

        # neighbors padded grid: 1 = wall/OOB by default, then mark interior.
        nbr_pad = np.ones((self.size + 2 * pad, self.size + 2 * pad), dtype=int)
        nbr_pad[pad:pad + self.size, pad:pad + self.size] = 0
        if self._obstacle_set:
            obs_arr = np.array(list(self._obstacle_set), dtype=int)
            nbr_pad[obs_arr[:, 0] + pad, obs_arr[:, 1] + pad] = 1
        if self.visited:
            vis_arr = np.array(list(self.visited), dtype=int)
            vx = vis_arr[:, 0] + pad
            vy = vis_arr[:, 1] + pad
            not_obs = nbr_pad[vx, vy] != 1
            nbr_pad[vx[not_obs], vy[not_obs]] = 2

        # visited_neighbors padded grid: 0 by default (OOB or non-visited),
        # 1 only where the agent has visited.
        vis_pad = np.zeros((self.size + 2 * pad, self.size + 2 * pad), dtype=int)
        if self.visited:
            vis_arr = np.array(list(self.visited), dtype=int)
            vis_pad[vis_arr[:, 0] + pad, vis_arr[:, 1] + pad] = 1

        cx, cy = ax + pad, ay + pad
        self._neighbors = nbr_pad[cx - pad:cx + pad + 1, cy - pad:cy + pad + 1].T
        self._visited_neighbors = vis_pad[cx - pad:cx + pad + 1, cy - pad:cy + pad + 1].T

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
            "neighbors": self._neighbors.astype(np.float32),
            "visited_neighbors": self._visited_neighbors.astype(np.float32),
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
        # Only mask out-of-bounds moves; obstacles are discovered, not masked.
        ax, ay = self._agent_location
        mask = np.zeros(4, dtype=bool)
        for action, direction in self._action_to_direction.items():
            nx, ny = ax + direction[0], ay + direction[1]
            if 0 <= nx < self.size and 0 <= ny < self.size:
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
        self.consecutive_revisits = 0
        self._milestones_awarded = set()

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
        self._build_windows(self._visited_grid())

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
        stayed_in_place = np.array_equal(self._agent_location, old_location)

        # Reward shaping (v3.2)
        reward = -0.05
        if stayed_in_place:
            reward -= 0.5
            self.consecutive_revisits += 1
        elif is_new_cell:
            reward += 1.0
            self.visited.add(current_pos)
            self.consecutive_revisits = 0
        else:
            reward -= 0.25
            self.consecutive_revisits += 1

        # Partial-coverage milestones — each awarded once per episode.
        cov = self.coverage_ratio
        for m in (0.25, 0.50, 0.75):
            if cov >= m and m not in self._milestones_awarded:
                reward += 2.0
                self._milestones_awarded.add(m)

        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage
        truncated = self.count_steps >= self.max_steps and not terminated

        if full_coverage:
            reward += 10.0 * (self.size / 5.0)

        # Refresh windows for the next observation.
        self._build_windows(self._visited_grid())

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
