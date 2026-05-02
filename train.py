"""
train.py — PPO training with domain randomization for FABS Track 2.

Trains ONLY on maps/train*.npy (test maps are held out for benchmarking).
Domain randomization (random start cell + small physics jitter on each
reset) is enabled at train time to improve generalization to held-out
and judges' OOD maps.

Usage:
    python train.py                        # default: 500k steps on T4 ~ 1.5h
    python train.py --steps 1000000 --n-envs 8
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from env import Maze3DEnv


class ProceduralMazeEnv(gym.Env):
    """Generates a fresh maze on every reset using maze_gen.generate_map.

    Why: 8 fixed train maps left enough specific topologies un-mastered that
    the policy still failed on holdouts. Procedural generation gives the
    agent infinite map variety in the same physics distribution mix
    (in-dist / OOD) — true generalization rather than coverage hoping.
    """

    metadata = Maze3DEnv.metadata

    def __init__(self, size: int = 9, ood_prob: float = 0.5, randomize: bool = True, seed: int = 0):
        super().__init__()
        from maze_gen import generate_map
        self._gen = generate_map
        self._size = size
        self._ood_prob = ood_prob
        self._randomize = randomize
        self._rng = np.random.default_rng(seed)
        # Build the underlying env with a throwaway initial maze
        m0 = generate_map(size, seed=int(self._rng.integers(1 << 30)), ood=False)
        self._env = Maze3DEnv(map_data=m0, randomize=randomize, seed=seed)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        is_ood = bool(self._rng.random() < self._ood_prob)
        m = self._gen(self._size, seed=int(self._rng.integers(1 << 30)), ood=is_ood)
        self._env.reload(m)
        return self._env.reset(seed=int(self._rng.integers(1 << 30)), options=options)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class MapShuffleEnv(gym.Env):
    """Picks a random map from `map_paths` on every reset (fixed-pool baseline)."""

    metadata = Maze3DEnv.metadata

    def __init__(self, map_paths: list[str], randomize: bool, seed: int = 0):
        super().__init__()
        if not map_paths:
            raise ValueError("map_paths must be non-empty")
        self._envs = {p: Maze3DEnv(p, randomize=randomize, seed=seed) for p in map_paths}
        self._paths = list(map_paths)
        self._rng = np.random.default_rng(seed)
        self.current = self._envs[self._paths[0]]
        self.action_space = self.current.action_space
        self.observation_space = self.current.observation_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        path = self._paths[int(self._rng.integers(len(self._paths)))]
        self.current = self._envs[path]
        return self.current.reset(seed=int(self._rng.integers(1 << 30)), options=options)

    def step(self, action):
        return self.current.step(action)

    def render(self):
        return self.current.render()

    def close(self):
        for e in self._envs.values():
            e.close()


def make_env(map_paths: list[str], randomize: bool, rank: int, seed: int = 0,
             procedural: bool = False, size: int = 9):
    def _init():
        if procedural:
            env = ProceduralMazeEnv(size=size, ood_prob=0.5, randomize=randomize,
                                    seed=seed + rank)
        else:
            env = MapShuffleEnv(map_paths=map_paths, randomize=randomize, seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maps-dir", type=str, default="maps")
    parser.add_argument("--out-dir", type=str, default="agents")
    parser.add_argument("--tb", type=str, default="tb")
    parser.add_argument("--no-dr", action="store_true",
                        help="disable Domain Randomization (random start cell). "
                             "On by default with BFS reward shaping, since DR helps "
                             "generalization without distorting the reward signal.")
    parser.add_argument("--procedural", action="store_true",
                        help="generate fresh mazes during training instead of "
                             "shuffling fixed maps/train*.npy.")
    parser.add_argument("--lstm", action="store_true",
                        help="use RecurrentPPO with LSTM policy. The hidden state "
                             "lets the agent remember where it has been and avoid "
                             "loops in dead-ends — essential for solving every "
                             "topology in a fixed pool, not just the lucky ones.")
    parser.add_argument("--subproc", action="store_true",
                        help="use SubprocVecEnv instead of DummyVecEnv")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tb).mkdir(parents=True, exist_ok=True)

    train_maps = sorted(glob.glob(os.path.join(args.maps_dir, "train*.npy")))
    test_maps = sorted(glob.glob(os.path.join(args.maps_dir, "test*.npy")))
    if not train_maps:
        raise FileNotFoundError(
            f"No train*.npy in {args.maps_dir}. Run `python maze_gen.py` first."
        )
    print(f"[train] {len(train_maps)} train maps, {len(test_maps)} test maps")
    print(f"[train] device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    randomize = not args.no_dr
    procedural = args.procedural
    print(f"[train] procedural={procedural} dr={randomize} lstm={args.lstm} train_maps={len(train_maps)}")
    if args.lstm and RecurrentPPO is None:
        raise RuntimeError("--lstm requires sb3-contrib. Run `pip install sb3-contrib==2.3.0`.")
    env_fns = [
        make_env(train_maps, randomize=randomize, rank=i, seed=args.seed,
                 procedural=procedural)
        for i in range(args.n_envs)
    ]
    VecCls = SubprocVecEnv if args.subproc else DummyVecEnv
    vec_env = VecCls(env_fns)

    # Eval: single train map, no DR — for EvalCallback's "best_model" tracking
    def _eval_init():
        env = Maze3DEnv(train_maps[0], randomize=False, seed=args.seed + 999)
        env = Monitor(env)
        return env
    eval_env = DummyVecEnv([_eval_init])

    if args.lstm:
        # RecurrentPPO with LSTM. The hidden state acts as implicit episodic
        # memory: the policy can "remember" cells it has tried, dead-ends it
        # bounced off, etc. — exactly the missing piece for solving every
        # topology in the train pool, not just the easy ones. Hyperparams are
        # smaller n_steps (LSTM unroll cost) and higher ent_coef (the wider
        # state space needs more exploration).
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[256, 256],
                lstm_hidden_size=128,
            ),
            tensorboard_log=args.tb,
            verbose=1,
            seed=args.seed,
            device="auto",
        )
    else:
        # Memoryless PPO baseline.
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=args.tb,
            verbose=1,
            seed=args.seed,
            device="auto",
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.steps // (args.n_envs * 5), 1000),
            save_path=args.out_dir,
            name_prefix="ppo_ckpt",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=args.out_dir,
            log_path=args.tb,
            eval_freq=max(args.steps // (args.n_envs * 10), 1000),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(args.out_dir, "agent"))
    print(f"[train] saved {args.out_dir}/agent.zip")


if __name__ == "__main__":
    main()
