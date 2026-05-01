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
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


N_RAYS = 8
N_OBS = 23
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
        map_path: str,
        max_steps: int = 250,
        render_mode: str | None = None,
        randomize: bool = False,
        seed: int | None = None,
    ):
        """
        Args:
            map_path: path to .npy file (see module docstring)
            max_steps: episode timeout (spec caps scoring at 200; 250 gives buffer)
            render_mode: 'human' | 'rgb_array' | None
            randomize: if True, randomize start cell + small physics jitter on
                each reset() — used for domain randomization during training
                to improve generalization to held-out / OOD maps.
            seed: RNG seed for randomize=True
        """
        super().__init__()
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f"Map file not found: {map_path}")
        self.map_path = map_path
        self._load(map_path)

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
        m = _load_map(path)
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

    def _random_open_cell(self) -> tuple[int, int]:
        h, w = self.grid.shape
        for _ in range(200):
            r = int(self._rng.integers(1, h - 1))
            c = int(self._rng.integers(1, w - 1))
            if self.grid[r, c] == 0 and (r, c) != self.goal:
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
            r, c = self._random_open_cell()
            self.pos = np.array([r, c], dtype=np.float32) + 0.5
            self._physics_scale = float(self._rng.uniform(0.9, 1.1))
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
        """Cast N_RAYS rays from agent. Returns (norm_distances, surface_codes)."""
        angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
        max_dist = float(np.hypot(*self.grid.shape))
        step = 0.2
        dists = np.zeros(N_RAYS, dtype=np.float32)
        codes = np.zeros(N_RAYS, dtype=np.float32)
        for i, ang in enumerate(angles):
            dr, dc = np.sin(ang), np.cos(ang)
            d = 0.0
            r = c = 0
            while d < max_dist:
                d += step
                r = int(self.pos[0] + dr * d)
                c = int(self.pos[1] + dc * d)
                if self._is_wall(r, c):
                    break
            dists[i] = min(d / max_dist, 1.0)
            r_safe = int(np.clip(r, 0, self.grid.shape[0] - 1))
            c_safe = int(np.clip(c, 0, self.grid.shape[1] - 1))
            codes[i] = _surface_code(float(self.surface[r_safe, c_safe]))
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
        # Layout: 8 dist + 8 surface + 2 goal_dxy + 3 (fric, slope_avg, temp) + 2 vel = 23
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
        goal_xy = np.array(self.goal, dtype=np.float32) + 0.5
        prev_dist = float(np.linalg.norm(prev_pos - goal_xy))
        cur_dist = float(np.linalg.norm(self.pos - goal_xy))
        progress = (prev_dist - cur_dist) / self._diag
        fric_bonus = 1.0 if self.surface[rr, cc] >= 1.0 else 0.0

        reward = -0.05 + 1.5 * progress + 0.05 * fric_bonus
        if wall_hit:
            reward -= 0.3

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
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
        plt.show()
        plt.close(fig)
        return None

    def close(self):
        pass
