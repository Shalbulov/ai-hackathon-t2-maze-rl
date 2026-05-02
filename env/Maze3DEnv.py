"""
Maze3DEnv — Gymnasium environment for the Track 2 hackathon maze task.

Observation (Box, 27): 8 ray-distances, 8 ray-surfaces, 2 goal-direction,
3 (friction, slope, temperature), 2 velocity, 4 neighbor visit-counts.
Action (Box, 2): continuous force in [-1, 1]^2.

Surfaces (friction): asphalt 1.0, grass 0.7, sand 0.4, ice 1.2.
Temperature: cold gives extra energy, heat caps speed.

Map .npy schema:
    grid:     int8   [H, W]      1=wall, 0=open
    surface:  float  [H, W]      one of {1.0, 0.7, 0.4, 1.2}
    slope:    float  [H, W, 2]   each in [-1, 1]
    temp:     float  [H, W]      in [0, 1]
    start:    (row, col)
    goal:     (row, col)
"""
from __future__ import annotations

import os
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


N_RAYS = 8
N_OBS_BASE = 23
N_OBS_VISIT = 4
N_OBS = N_OBS_BASE + N_OBS_VISIT  # 27
VISIT_SATURATION = 5.0
SURFACE_VALUES = np.array([0.4, 0.7, 1.0, 1.2], dtype=np.float32)

# Terminal speed v_term = action * fric * DT / (1 - damping) = 2.2 cells/step.
DT = 0.22
MAX_VEL = 2.5
BASE_DAMPING = 0.90
SLOPE_PULL = 0.35


def _load_map(path: str) -> dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return obj.item()
    return dict(obj)


def _surface_code(value: float) -> float:
    """Map a surface friction value to a normalized code in [0, 1]."""
    idx = int(np.argmin(np.abs(SURFACE_VALUES - value)))
    return idx / (len(SURFACE_VALUES) - 1)


class Maze3DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        map_path: str | None = None,
        max_steps: int = 250,
        render_mode: str | None = None,
        randomize: bool = False,
        seed: int | None = None,
        map_data: dict | None = None,
    ):
        """
        Args:
            map_path: path to .npy file (see module docstring for schema)
            map_data: alternative to map_path — map dict in memory
            max_steps: episode timeout
            render_mode: 'human' | 'rgb_array' | None
            randomize: random start cell + small physics jitter each reset
            seed: RNG seed
        """
        super().__init__()
        if map_data is not None:
            self.map_path = "<inline>"
            self._load_dict(map_data)
        elif map_path is not None:
            if not os.path.isfile(map_path):
                raise FileNotFoundError(f"Map file not found: {map_path}")
            self.map_path = map_path
            self._load(map_path)
        else:
            raise ValueError("Maze3DEnv needs either map_path or map_data")

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.randomize = randomize

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_OBS,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self.pos = np.array(self.start, dtype=np.float32) + 0.5
        self.vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.visits = np.zeros_like(self.grid, dtype=np.int32)
        self.trajectory: list[tuple[float, float]] = [tuple(self.pos.tolist())]

        diag = float(np.hypot(*self.grid.shape))
        self._diag = diag if diag > 0 else 1.0
        self._physics_scale = 1.0

    def _load(self, path: str) -> None:
        self._load_dict(_load_map(path))

    def _load_dict(self, m: dict) -> None:
        self.grid = np.asarray(m["grid"], dtype=np.int8)
        self.surface = np.asarray(m["surface"], dtype=np.float32)
        self.slope = np.asarray(m["slope"], dtype=np.float32)
        self.temp = np.asarray(m["temp"], dtype=np.float32)
        self.start = tuple(map(int, m["start"]))
        self.goal = tuple(map(int, m["goal"]))
        h, w = self.grid.shape
        assert self.surface.shape == (h, w)
        assert self.slope.shape == (h, w, 2)
        assert self.temp.shape == (h, w)
        # BFS-distance grid from goal — drives reward shaping. Euclidean
        # distance shrinks when an agent moves into a wall toward the goal,
        # so the policy learns to bash walls. BFS only shrinks on real
        # progress along a reachable path.
        self.bfs_dist = self._bfs_from_goal()
        if not hasattr(self, "visits") or self.visits.shape != self.grid.shape:
            self.visits = np.zeros_like(self.grid, dtype=np.int32)

    def reload(self, map_data: dict) -> None:
        """Hot-swap to a new map without re-instantiation."""
        self._load_dict(map_data)
        diag = float(np.hypot(*self.grid.shape))
        self._diag = diag if diag > 0 else 1.0

    def _bfs_from_goal(self) -> np.ndarray:
        h, w = self.grid.shape
        d = np.full((h, w), 1e6, dtype=np.float32)
        d[self.goal] = 0.0
        q: deque[tuple[int, int]] = deque([self.goal])
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and self.grid[nr, nc] == 0:
                    if d[nr, nc] > d[r, c] + 1:
                        d[nr, nc] = d[r, c] + 1
                        q.append((nr, nc))
        return d

    def _random_open_cell(self) -> tuple[int, int]:
        """Random open cell with Manhattan distance ≥ 4 from goal."""
        h, w = self.grid.shape
        gr, gc = self.goal
        for _ in range(200):
            r = int(self._rng.integers(1, h - 1))
            c = int(self._rng.integers(1, w - 1))
            if self.grid[r, c] == 0 and (r, c) != self.goal:
                if abs(r - gr) + abs(c - gc) >= 4:
                    return r, c
        return self.start

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.randomize:
            # 50% canonical start, 50% random open cell (domain randomization)
            if self._rng.random() < 0.5:
                self.pos = np.array(self.start, dtype=np.float32) + 0.5
            else:
                r, c = self._random_open_cell()
                self.pos = np.array([r, c], dtype=np.float32) + 0.5
            self._physics_scale = float(self._rng.uniform(0.95, 1.05))
        else:
            self.pos = np.array(self.start, dtype=np.float32) + 0.5
            self._physics_scale = 1.0

        self.vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.visits.fill(0)
        self.trajectory = [(float(self.pos[0]), float(self.pos[1]))]
        return self._get_obs(), self._info()

    def _cell(self, pos: np.ndarray) -> tuple[int, int]:
        h, w = self.grid.shape
        r = int(np.clip(pos[0], 0, h - 1))
        c = int(np.clip(pos[1], 0, w - 1))
        return r, c

    def _is_wall(self, r: int, c: int) -> bool:
        h, w = self.grid.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return True
        return bool(self.grid[r, c])

    def _raycast(self) -> tuple[np.ndarray, np.ndarray]:
        """Cast N_RAYS rays from the agent (vectorized over rays × samples)."""
        h, w = self.grid.shape
        max_dist = float(np.hypot(h, w))
        n_samples = int(max_dist / 0.2) + 1
        d_samples = np.arange(1, n_samples + 1, dtype=np.float32) * 0.2  # (S,)

        if not hasattr(self, "_ray_sin") or self._ray_sin.shape[0] != N_RAYS:
            angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
            self._ray_sin = np.sin(angles).astype(np.float32)[:, None]  # (R, 1)
            self._ray_cos = np.cos(angles).astype(np.float32)[:, None]

        rs = (self.pos[0] + self._ray_sin * d_samples).astype(np.int32)  # (R, S)
        cs = (self.pos[1] + self._ray_cos * d_samples).astype(np.int32)
        out_of_bounds = (rs < 0) | (rs >= h) | (cs < 0) | (cs >= w)
        rs_safe = np.clip(rs, 0, h - 1)
        cs_safe = np.clip(cs, 0, w - 1)
        is_wall = out_of_bounds | (self.grid[rs_safe, cs_safe] == 1)

        first_hit = np.argmax(is_wall, axis=1)  # (R,)
        no_hit = ~is_wall.any(axis=1)
        first_hit = np.where(no_hit, n_samples - 1, first_hit)
        hit_d = d_samples[first_hit]
        dists = np.minimum(hit_d / max_dist, 1.0).astype(np.float32)

        ray_idx = np.arange(N_RAYS)
        hit_r = rs_safe[ray_idx, first_hit]
        hit_c = cs_safe[ray_idx, first_hit]
        surf = self.surface[hit_r, hit_c]  # (R,)
        diffs = np.abs(SURFACE_VALUES[None, :] - surf[:, None])
        codes = (np.argmin(diffs, axis=1) / (len(SURFACE_VALUES) - 1)).astype(np.float32)
        return dists, codes

    def _get_obs(self) -> np.ndarray:
        dists, codes = self._raycast()
        r, c = self._cell(self.pos)
        slope_xy = self.slope[r, c]
        fric = float(self.surface[r, c])
        temp = float(self.temp[r, c])
        h, w = self.grid.shape
        gdx = (self.goal[0] + 0.5 - self.pos[0]) / h
        gdy = (self.goal[1] + 0.5 - self.pos[1]) / w

        # Neighbor visit counts (N/E/S/W), saturated to 1.0; walls report 1.0
        # so the policy treats unreachable directions as "stale".
        def _vc(rr: int, cc: int) -> float:
            if 0 <= rr < h and 0 <= cc < w and self.grid[rr, cc] == 0:
                return min(self.visits[rr, cc] / VISIT_SATURATION, 1.0)
            return 1.0
        visit_neighbors = np.array(
            [_vc(r - 1, c), _vc(r + 1, c), _vc(r, c - 1), _vc(r, c + 1)],
            dtype=np.float32,
        )

        # 8 dist + 8 surface + 2 goal_dxy + 3 (fric, slope, temp) + 2 vel + 4 visits = 27
        obs = np.concatenate(
            [
                dists,
                codes,
                np.array(
                    [np.clip(gdx, -1.0, 1.0), np.clip(gdy, -1.0, 1.0)],
                    dtype=np.float32,
                ),
                np.array(
                    [(fric - 0.8) / 0.4, float((slope_xy[0] + slope_xy[1]) * 0.5), temp],
                    dtype=np.float32,
                ),
                np.clip(self.vel / MAX_VEL, -1.0, 1.0),
                visit_neighbors,
            ]
        ).astype(np.float32)
        assert obs.shape == (N_OBS,), f"obs shape {obs.shape}"
        return obs

    def _info(self) -> dict[str, Any]:
        return {
            "pos": tuple(self.pos.tolist()),
            "vel": tuple(self.vel.tolist()),
            "steps": self.steps,
            "is_success": tuple(self._cell(self.pos)) == self.goal,
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        r, c = self._cell(self.pos)
        fric = float(self.surface[r, c]) * self._physics_scale
        slope = self.slope[r, c]
        temp = float(self.temp[r, c])

        # Temperature regimes per spec: cold → +energy, heat → speed cap
        if temp < 0.33:
            energy, speed_mul = 1.5, 1.0   # cold
        elif temp > 0.66:
            energy, speed_mul = 1.0, 0.6   # heat
        else:
            energy, speed_mul = 1.0, 1.0   # normal

        # Continuous physics: vel += (action·fric·energy + slope_pull) · dt
        self.vel += (action * fric * energy + slope * SLOPE_PULL) * DT
        self.vel *= BASE_DAMPING * speed_mul
        self.vel = np.clip(self.vel, -MAX_VEL, MAX_VEL)

        prev_pos = self.pos.copy()
        new_pos = self.pos + self.vel * DT

        # Sliding collision: per-axis test so the agent slides along walls
        wall_hit = False
        nr, nc = self._cell(np.array([new_pos[0], self.pos[1]]))
        if self._is_wall(nr, nc):
            new_pos[0] = self.pos[0]
            self.vel[0] = -0.3 * self.vel[0]
            wall_hit = True
        nr, nc = self._cell(np.array([new_pos[0], new_pos[1]]))
        if self._is_wall(nr, nc):
            new_pos[1] = self.pos[1]
            self.vel[1] = -0.3 * self.vel[1]
            wall_hit = True

        self.pos = new_pos
        self.steps += 1
        rr, cc = self._cell(self.pos)
        self.visits[rr, cc] += 1
        self.trajectory.append((float(self.pos[0]), float(self.pos[1])))

        # Reward = time penalty + BFS-progress + first-visit bonus
        #        + friction bonus − wall penalty + terminal goal bonus.
        fric_bonus = 1.0 if self.surface[rr, cc] >= 1.0 else 0.0

        prev_r, prev_c = self._cell(prev_pos)
        bfs_progress = float(self.bfs_dist[prev_r, prev_c]) - float(self.bfs_dist[rr, cc])

        first_visit_bonus = 0.1 if self.visits[rr, cc] == 1 else 0.0

        reward = -0.02 + 1.0 * bfs_progress + 0.02 * fric_bonus + first_visit_bonus
        if wall_hit:
            reward -= 0.2

        terminated = False
        truncated = False
        if (rr, cc) == self.goal:
            reward += 50.0
            terminated = True
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.grid, cmap="binary", origin="upper", interpolation="nearest")
        ax.imshow(self.surface, cmap="viridis", alpha=0.35, origin="upper")
        ax.scatter([self.pos[1]], [self.pos[0]], c="red", s=120, zorder=4, label="agent")
        ax.scatter([self.goal[1]], [self.goal[0]], c="lime", marker="*", s=240, zorder=4, label="goal")
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 1], traj[:, 0], "r-", linewidth=1.2, alpha=0.6)
        ax.set_title(f"step={self.steps}  vel=({self.vel[0]:+.2f},{self.vel[1]:+.2f})")
        ax.legend(loc="upper right")
        ax.axis("off")
        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            plt.close(fig)
            return img
        plt.show()
        plt.close(fig)
        return None

    def close(self):
        pass
