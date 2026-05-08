from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# Coverage Path Planning (CPP) environment — partial-observability compliant,
# 5x5-windowed (v3.7 / minimalist + nearest-unvisited compass).
#
# Observation (v3.7):
#   - "agent": 10 floats
#       [0,1]  pose_x, pose_y       (normalized [0,1])
#       [2]    coverage_ratio       (visited / total_free, [0,1])
#       [3-6]  4 directional unvisited ratios (right/up/left/down, [0,1])
#       [7,8]  signed (dx, dy) toward nearest UNVISITED cell that the
#              agent does NOT already know is an obstacle. Normalized
#              by (size-1) so values are in [-1, 1].
#       [9]    Manhattan distance to that nearest cell, normalized by
#              2*(size-1), in [0, 1].
#   - "neighbors": 5x5 sensor view centred on the agent
#       (0 = free, 1 = obstacle/wall, 2 = visited).
#
# v3.7 motivation: v3.6 (7-feature agent vec) reached 67% stoch full
# coverage on 10x10 and 0% on 20x20 because the directional ratios
# decay to ~1/N when only one cell is left, providing essentially no
# signal for endgame closure. The new (dx, dy, dist) compass to the
# nearest cell the agent has NOT visited and NOT seen-as-obstacle gives
# a strong, distance-weighted vector that points the agent at unfinished
# work — exactly the missing signal. It is computed only from the
# agent's own visited set and the obstacles it has revealed via the 5x5
# sensor, so partial observability is preserved (no global obstacle map
# leakage like v3.5's visited_map). This is feature engineering, not
# planning — argmin over Manhattan distance, no graph search.
#
# v3.7 also adds a solvability filter at reset(): obstacle layouts that
# enclose any free cell (making full_coverage objectively unreachable)
# are regenerated until a solvable layout is produced. This was approved
# by the professor (08/05/2026) provided it lives at the generation step
# and is documented. Without it, ~25-35% of 10x10 layouts had unreachable
# pockets that capped success rate regardless of policy quality.
#
# This still matches the assignment text ("matrix 3x3 or 5x5 with the
# agent at the center") — the central observation is the 5x5 sensor.
# The agent vector contains derived quantities from "other information
# collected during exploration" (visited set + sensor-revealed obstacles).
#
# Action masking (used by MaskablePPO) only masks moves that would leave the
# grid. Obstacles are NOT masked: the agent must discover them through the
# 5x5 sensor and the stuck penalty.
#
# Reward (v3.4 — v3.2 base + potential-based shaping):
#   step base               -0.05
#   new cell                +1.0
#   revisit                 -0.25
#   stuck (vs obstacle)     -0.5
#   25% / 50% / 75% milestone  +2.0 each  (one-time per episode)
#   full coverage           +10 * (size / 5)
#   truncation              0
#   potential shaping       +K * (γ * cov(s') - cov(s))   K=10, γ=0.997
#
# v3.3 (90/95/98 + proportional trunc penalty) was tried and reverted —
# it backfired by rewarding conservatism. v3.4 adds a *potential-based*
# shaping term derived from Ng, Harada & Russell (1999): the term
# γ·Φ(s') - Φ(s) is provably reward-invariant (same optimal policy as
# the unshaped problem) but produces denser gradient signal. With Φ = 10·
# coverage_ratio, the shaping is on the order of ~+0.03 per new-cell visit
# in early game and ~0 in endgame; it amplifies the "make progress" signal
# without removing pressure to fully close. Mathematically safe per Ng's
# theorem.
#

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    WINDOW = 5  # spatial window radius parameter (window is 2*PAD+1 = 5 wide)
    PAD = 2

    # Potential-based shaping constants (Ng, Harada, Russell 1999).
    # Φ(s) = SHAPING_K * coverage_ratio. F(s,s') = γ_shaping * Φ(s') - Φ(s).
    # Mathematically guarantees the optimal policy is unchanged, while
    # densifying the gradient signal toward higher coverage states.
    SHAPING_K = 10.0
    SHAPING_GAMMA = 0.997

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

        # Cells the agent has *seen* through its 5x5 sensor and confirmed are
        # obstacles. Kept so the nearest-unvisited compass can skip them; the
        # agent should not be told about obstacles it has not yet observed
        # (partial observability), but cells it has seen are fair game.
        self._known_obstacles: set = set()

        # Cached structures (NOT exposed to the agent).
        self._obstacle_set: set = set()
        self._obstacle_grid: np.ndarray = np.zeros((size, size), dtype=bool)

        # Pre-built coordinate arrays for vectorised _get_directional_signals().
        _r = np.arange(size)
        self._xs, self._ys = np.meshgrid(_r, _r, indexing='ij')

        # Agent vec layout:
        #   pose_x, pose_y        -> [0, 1]
        #   coverage              -> [0, 1]
        #   4 directional ratios  -> [0, 1]
        #   dx, dy compass        -> [-1, 1]
        #   dist compass          -> [0, 1]
        agent_low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            dtype=np.float32,
        )
        agent_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=agent_low,
                high=agent_high,
                dtype=np.float32,
            ),
            "neighbors": gym.spaces.Box(
                low=np.zeros((self.WINDOW, self.WINDOW), dtype=np.float32),
                high=np.full((self.WINDOW, self.WINDOW), 2.0, dtype=np.float32),
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
        """Compute the 5x5 neighbors window via padded slicing.

        Encoding: 0 = free / unvisited, 1 = obstacle / OOB wall, 2 = visited.
        Also expands `self._known_obstacles` with any obstacle cells revealed
        inside the current sensor window (partial-observability respected).
        """
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

        cx, cy = ax + pad, ay + pad
        self._neighbors = nbr_pad[cx - pad:cx + pad + 1, cy - pad:cy + pad + 1].T

        # Expand known_obstacles with any obstacle cells inside the 5x5
        # sensor window (in world coordinates). Iterating self._obstacle_set
        # is O(K) where K = obstacle count (3-48), much cheaper than scanning
        # the 25-cell window each step.
        x_lo, x_hi = ax - pad, ax + pad
        y_lo, y_hi = ay - pad, ay + pad
        for (ox, oy) in self._obstacle_set:
            if x_lo <= ox <= x_hi and y_lo <= oy <= y_hi:
                self._known_obstacles.add((ox, oy))

    def _get_nearest_unvisited(self, visited_grid: np.ndarray):
        """Return (dx, dy, dist) toward the nearest cell the agent has
        neither visited nor confirmed is an obstacle. Computed only from
        agent-known information so partial observability is preserved.

        Output ranges: dx, dy in [-1, 1]; dist in [0, 1].
        Returns (0, 0, 0) when nothing remains (full coverage / endgame).
        """
        ax, ay = self._agent_location
        candidate = ~visited_grid
        if self._known_obstacles:
            ko_arr = np.array(list(self._known_obstacles), dtype=int)
            candidate[ko_arr[:, 0], ko_arr[:, 1]] = False
        if not candidate.any():
            return 0.0, 0.0, 0.0
        dist = np.abs(self._xs - ax) + np.abs(self._ys - ay)
        # Use a large sentinel for ineligible cells so argmin lands on a
        # candidate. 9999 is > any possible Manhattan distance up to 20x20.
        dist_inf = np.where(candidate, dist, 9999)
        flat_idx = int(np.argmin(dist_inf))
        nx, ny = np.unravel_index(flat_idx, dist_inf.shape)
        norm = max(self.size - 1, 1)
        dx = float((int(nx) - ax) / norm)
        dy = float((int(ny) - ay) / norm)
        d = float((abs(int(nx) - ax) + abs(int(ny) - ay)) / (2 * norm))
        return dx, dy, d

    def _get_obs(self):
        visited_grid = self._visited_grid()
        dir_signals = self._get_directional_signals(visited_grid)
        ndx, ndy, ndd = self._get_nearest_unvisited(visited_grid)
        return {
            "agent": np.array([
                self._agent_location[0] / max(self.size - 1, 1),
                self._agent_location[1] / max(self.size - 1, 1),
                self.coverage_ratio,
                dir_signals[0],
                dir_signals[1],
                dir_signals[2],
                dir_signals[3],
                ndx,
                ndy,
                ndd,
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

    def _is_layout_solvable(self) -> bool:
        """BFS reachability check from the agent's starting cell. Returns
        True iff every free (non-obstacle) cell is reachable, i.e. there
        are no enclosed pockets that would make full_coverage objectively
        unreachable regardless of policy quality.

        Used ONLY at reset/generation time, never for policy decisions.
        Professor approved this filter on 08/05/2026 (in-class) provided
        it is documented and lives at the environment-generation step.
        """
        from collections import deque
        start = tuple(int(c) for c in self._agent_location)
        seen = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + ddx, y + ddy
                if (0 <= nx < self.size and 0 <= ny < self.size
                        and (nx, ny) not in self._obstacle_set
                        and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        return len(seen) == self.total_free_cells

    # Cap on retries to avoid pathological infinite loops in degenerate
    # configurations (very rare; typical 10x10/12-obstacle expects 1-3 retries).
    MAX_LAYOUT_RETRIES = 200

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.visited = set()
        self.consecutive_revisits = 0
        self._milestones_awarded = set()
        self._known_obstacles = set()

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Generate obstacles repeatedly until the layout is solvable
        # (every free cell reachable from agent start). Without this filter
        # ~25-35% of 10x10 layouts (and most 20x20) randomly enclose a cell,
        # capping full_coverage at <100% no matter what policy is learned.
        for _retry in range(self.MAX_LAYOUT_RETRIES):
            self.obstacles_locations = []
            _tmp_obs_set: set = {tuple(self._agent_location)}
            for _ in range(self.obs_quantity):
                obstacle_location = self._agent_location.copy()
                while tuple(obstacle_location) in _tmp_obs_set:
                    obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.obstacles_locations.append(obstacle_location)
                _tmp_obs_set.add(tuple(obstacle_location))
            self._rebuild_obstacle_caches()
            if self._is_layout_solvable():
                break
        # If we exhausted MAX_LAYOUT_RETRIES we fall through with the last
        # generated layout — vanishingly unlikely with 12-48 obstacles.

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

        # Capture potential before updating visited set so we can compute
        # the potential-based shaping term F = γ·Φ(s') - Φ(s) below.
        phi_before = self.SHAPING_K * self.coverage_ratio

        # Reward shaping (v3.4)
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

        # Potential-based shaping: F = γ·Φ(s') - Φ(s). With Φ = K·coverage,
        # this is provably reward-invariant (Ng, Harada, Russell 1999) — it
        # does not change the optimal policy, but it gives a small dense
        # gradient signal at every step toward higher-coverage states.
        phi_after = self.SHAPING_K * self.coverage_ratio
        reward += self.SHAPING_GAMMA * phi_after - phi_before

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
