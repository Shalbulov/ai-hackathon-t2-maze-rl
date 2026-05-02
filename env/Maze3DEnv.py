"""
Maze3DEnv — Custom Gymnasium environment for the Build-with-AI 2026 DeepTech
Hackathon Track 2 (FABS / GDG April): RL agent navigating a 3D-physics maze.

Spec (per https://fabstoryai.com/gdgapril):
    Observation: Box(23) — 8 ray-distances + 8 ray-surfaces + 2 goal_dxy
                 + 3 (friction, slope, temperature) + 2 velocity
    Action:      Box(2) ∈ [-1, 1]^2 — continuous force [fx, fy]
    Surfaces:    asphalt 1.0, grass 0.7, sand 0.4, ice 1.2
    Temperature: cold→energy ×1.5, normal ×1.0, heat→speed ×0.6
    Slope:       uphills slow, downhills accelerate

Physics is tuned for ~10×10 grids so that an optimal trajectory fits in
≤60 environment steps (the "Perfect" tier per the spec).

Map .npy file (allow_pickle=True):
    {
        'grid':    np.int8   [H, W]   1=wall, 0=open
        'surface': np.float32 [H, W]   ∈ {1.0, 0.7, 0.4, 1.2}
        'slope':   np.float32 [H, W, 2] each ∈ [-1, 1]   (slope_x, slope_y)
        'temp':    np.float32 [H, W]   ∈ [0, 1]   (0=cold, 0.5=normal, 1=heat)
        'start':   tuple[int, int]
        'goal':    tuple[int, int]
    }
"""
from __future__ import annotations

import os
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


N_RAYS = 8
N_OBS_BASE = 23   # spec-mandated features (8 dist + 8 surf + 2 goal + 3 phys + 2 vel)
N_OBS_VISIT = 4   # extension: visit counts of N/E/S/W neighbors (loop avoidance)
N_OBS = N_OBS_BASE + N_OBS_VISIT  # 27
VISIT_SATURATION = 5.0  # visits >= 5 saturate to 1.0 in obs
SURFACE_VALUES = np.array([0.4, 0.7, 1.0, 1.2], dtype=np.float32)  # sand, grass, asphalt, ice

# Physics constants — tuned so 9×9 mazes (path ~16-19 cells) hit the rubric's
# "Perfect ≤60 env-steps" tier. Terminal speed for full-throttle on asphalt:
#   v_term = (action * fric * dt) / (1 - damping) = 0.22 / 0.10 = 2.2 cells/step
# Per-step displacement at v_term: 2.2 * 0.22 = 0.48 cells/env-step.
# A 17-cell shortest path with ~70% momentum efficiency ≈ 50 env-steps.
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
            map_path: path to .npy file (see module docstring)
            map_data: alternative to map_path — pass map dict directly (used
                by procedural training, where mazes are regenerated each reset)
            max_steps: episode timeout (spec caps scoring at 200; 250 gives buffer)
            render_mode: 'human' | 'rgb_array' | None
            randomize: if True, randomize start cell + small physics jitter on
                each reset() — used for domain randomization during training
                to improve generalization to held-out / OOD maps.
            seed: RNG seed for randomize=True
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
        self._physics_scale = 1.0  # mutated by randomize

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
        # BFS distance from goal for potential-based reward shaping. This is
        # critical: Euclidean distance can decrease even when the agent moves
        # INTO a wall (because the wall is between agent and goal). With
        # Euclidean shaping the policy learns to bash walls. BFS distance only
        # decreases when the agent is actually making progress along a
        # reachable path.
        self.bfs_dist = self._bfs_from_goal()
        # visits buffer must match current grid shape
        if not hasattr(self, "visits") or self.visits.shape != self.grid.shape:
            self.visits = np.zeros_like(self.grid, dtype=np.int32)

    def reload(self, map_data: dict) -> None:
        """Hot-swap to a fresh map (used by procedural training)."""
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
        """Pick a random open cell that is at least 4 cells from the goal.

        Why the distance floor: in a 9×9 maze, an unconstrained random start
        often lands adjacent to the goal, giving 1-step episodes that don't
        teach navigation. Forcing min-distance keeps every episode a real
        navigation task.
        """
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
            # Curriculum: 50% from canonical start (clean signal), 50% from
            # randomized start (generalization). Pure randomization drowned the
            # signal in noise and the policy collapsed to "stand still".
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
        """Cast N_RAYS rays from agent (vectorized). Returns (norm_distances, surface_codes).

        Vectorized over all rays AND all sample distances at once via NumPy
        broadcasting. ~6× faster than the Python-loop version, which dominated
        env step time (env step is called every PPO/RecurrentPPO transition).
        """
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

        # Visit counts of 4 cardinal neighbors, normalized to [0, 1]. Walls
        # report 1.0 (saturated) so the policy treats them like "stale" cells
        # — there's no reward signal pulling toward unreachable directions.
        def _vc(rr: int, cc: int) -> float:
            if 0 <= rr < h and 0 <= cc < w and self.grid[rr, cc] == 0:
                return min(self.visits[rr, cc] / VISIT_SATURATION, 1.0)
            return 1.0
        visit_neighbors = np.array(
            [_vc(r - 1, c), _vc(r + 1, c), _vc(r, c - 1), _vc(r, c + 1)],
            dtype=np.float32,
        )

        # Layout: 8 dist + 8 surface + 2 goal_dxy + 3 (fric, slope_avg, temp)
        #       + 2 vel + 4 neighbor_visits = 27
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

        # Temperature: cold gives more energy, heat slows down (per spec)
        if temp < 0.33:       # cold
            energy = 1.5
            speed_mul = 1.0
        elif temp > 0.66:     # heat
            energy = 1.0
            speed_mul = 0.6
        else:                  # normal
            energy = 1.0
            speed_mul = 1.0

        # Continuous physics — adapted from aweeraman/RL-continuous-control
        # vel += (action_force * fric * energy + slope_pull) * dt
        self.vel += (action * fric * energy + slope * SLOPE_PULL) * DT
        self.vel *= BASE_DAMPING * speed_mul
        self.vel = np.clip(self.vel, -MAX_VEL, MAX_VEL)

        prev_pos = self.pos.copy()
        new_pos = self.pos + self.vel * DT

        # Sliding collision: separate-axis test
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

        # Reward shaping (justified in README "Reward Design" section):
        # 1. small time penalty -> minimize steps
        # 2. dense progress signal -> avoids sparse-reward credit assignment trap
        # 3. surface bonus -> learn to prefer high-grip terrain
        # 4. wall penalty -> discourage scraping walls
        # 5. terminal goal bonus -> strong signal for the actual objective
        fric_bonus = 1.0 if self.surface[rr, cc] >= 1.0 else 0.0

        # Potential-based reward shaping using BFS distance (in cells).
        # bfs_progress > 0 ⇔ agent moved to a cell strictly closer to goal
        # along a reachable path. Euclidean shaping (the previous formula)
        # rewards moving INTO walls when the wall is between agent and goal,
        # which collapses learning. See _bfs_from_goal in this file.
        prev_r, prev_c = self._cell(prev_pos)
        prev_bfs = float(self.bfs_dist[prev_r, prev_c])
        cur_bfs = float(self.bfs_dist[rr, cc])
        bfs_progress = prev_bfs - cur_bfs  # +1 for one cell of true progress
        reward = -0.02 + 1.0 * bfs_progress + 0.02 * fric_bonus
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
            # tostring_rgb() removed in matplotlib 3.8+; buffer_rgba is the portable path
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            plt.close(fig)
            return img
        plt.show()
        plt.close(fig)
        return None

    def close(self):
        pass
