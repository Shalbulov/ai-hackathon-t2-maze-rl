"""
ablation.py — Component ablation study for FABS Track 2 rubric (8 pts).

Runs 3 short training runs with key components disabled, then compares
mean steps on test maps. This justifies which obs/reward components are
load-bearing.

Variants:
    full          — everything on (baseline)
    no_surface    — surface ray codes zeroed in obs
    no_progress   — only sparse reward (-0.05 per step + terminal goal)
    no_dr         — no domain randomization at training

Total runtime ~ 4 × (steps / 4) ≈ steps, so use a small step budget here
(e.g. 100k each) for the ablation table — the *main* agent stays in
agents/agent.zip from train.py.

Usage:
    python ablation.py --steps 100000
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import Maze3DEnv


class AblationWrapper(gym.Wrapper):
    """Selectively zero observation components or strip the dense reward."""

    def __init__(self, env, no_surface: bool = False, no_progress: bool = False):
        super().__init__(env)
        self.no_surface = no_surface
        self.no_progress = no_progress

    def _mask(self, obs):
        if self.no_surface:
            obs = obs.copy()
            obs[8:16] = 0.0
        return obs

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._mask(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if self.no_progress:
            # Strip the dense progress component: keep only step penalty and terminal bonus
            reward = -0.05 + (50.0 if term else 0.0)
        return self._mask(obs), reward, term, trunc, info


def make_env(map_path: str, randomize: bool, no_surface: bool, no_progress: bool, rank: int):
    def _init():
        env = Maze3DEnv(map_path=map_path, randomize=randomize, seed=rank)
        env = AblationWrapper(env, no_surface=no_surface, no_progress=no_progress)
        env = Monitor(env)
        return env
    return _init


def train_variant(name: str, train_maps, steps: int, *, no_surface=False, no_progress=False, no_dr=False, n_envs=4):
    print(f"\n=== variant: {name} ===")
    env_fns = [
        make_env(train_maps[i % len(train_maps)],
                 randomize=not no_dr,
                 no_surface=no_surface,
                 no_progress=no_progress,
                 rank=i)
        for i in range(n_envs)
    ]
    vec = DummyVecEnv(env_fns)
    model = PPO("MlpPolicy", vec, verbose=0, learning_rate=3e-4,
                n_steps=1024, batch_size=256, n_epochs=10,
                ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=steps, progress_bar=True)
    return model


def evaluate(model, paths, episodes: int, *, no_surface=False):
    means = []
    for p in paths:
        env = Maze3DEnv(map_path=p, randomize=False)
        env = AblationWrapper(env, no_surface=no_surface)
        steps_list = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
            steps_list.append(env.unwrapped.steps)
        means.append(float(np.mean(steps_list)))
    return float(np.mean(means))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--maps-dir", type=str, default="maps")
    parser.add_argument("--out", type=str, default="viz/ablation.txt")
    args = parser.parse_args()

    train_maps = sorted(glob.glob(os.path.join(args.maps_dir, "train*.npy")))
    test_maps = sorted(glob.glob(os.path.join(args.maps_dir, "test*.npy")))

    variants = {
        "full":        dict(),
        "no_surface":  dict(no_surface=True),
        "no_progress": dict(no_progress=True),
        "no_dr":       dict(no_dr=True),
    }

    results = {}
    for name, kw in variants.items():
        m = train_variant(name, train_maps, args.steps, **kw)
        train_steps = evaluate(m, train_maps, args.episodes, no_surface=kw.get("no_surface", False))
        test_steps = evaluate(m, test_maps, args.episodes, no_surface=kw.get("no_surface", False))
        results[name] = (train_steps, test_steps)
        print(f"  -> train={train_steps:.1f}  test={test_steps:.1f}")

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("variant        train_steps  test_steps  gap\n")
        f.write("-" * 50 + "\n")
        for name, (tr, te) in results.items():
            gap = (te - tr) / max(tr, 1e-6) * 100
            f.write(f"{name:<14} {tr:>10.1f}  {te:>10.1f}  {gap:+.1f}%\n")
    print(f"\nablation table written to {args.out}")


if __name__ == "__main__":
    main()
