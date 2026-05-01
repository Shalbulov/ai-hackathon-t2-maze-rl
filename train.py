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

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import Maze3DEnv


def make_env(map_path: str, randomize: bool, rank: int, seed: int = 0):
    def _init():
        env = Maze3DEnv(map_path=map_path, randomize=randomize, seed=seed + rank)
        env = Monitor(env)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maps-dir", type=str, default="maps")
    parser.add_argument("--out-dir", type=str, default="agents")
    parser.add_argument("--tb", type=str, default="tb")
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

    # Round-robin train maps across parallel envs, randomize=True for DR
    env_fns = [
        make_env(train_maps[i % len(train_maps)], randomize=True, rank=i, seed=args.seed)
        for i in range(args.n_envs)
    ]
    VecCls = SubprocVecEnv if args.subproc else DummyVecEnv
    vec_env = VecCls(env_fns)

    # Eval on a single train map without DR — used by EvalCallback for early-stop tracking
    eval_env = DummyVecEnv([make_env(train_maps[0], randomize=False, rank=999, seed=args.seed)])

    # PPO hyperparams chosen for short-horizon continuous control on small obs.
    # n_steps=1024 × 4 envs = 4096 transitions per rollout — good for PPO stability.
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
        ent_coef=0.02,             # raised: previous run collapsed to "stand still"
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
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
