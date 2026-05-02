"""
benchmark.py — Evaluate the trained agent on train + test maps and produce
all visualizations required by the FABS Track 2 rubric:

    viz/learning_curve.png       (from TB events)
    viz/heatmap_<map>.png        (visit-count 2D histogram)
    viz/trajectory_3d.gif        (3D pos over time)
    viz/before_after.gif         (untrained vs trained on test1)
    viz/results_table.txt        (mean ± std steps, per-map)

Usage:
    python benchmark.py                        # uses agents/agent.zip
    python benchmark.py --model agents/best_model.zip --episodes 100
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from env import Maze3DEnv


def load_model(path: str):
    """Try PPO first, fall back to RecurrentPPO. SB3 stores model class metadata
    in the zip but PPO.load() won't auto-route to the LSTM variant, so we sniff."""
    if path.endswith(".zip"):
        try:
            return PPO.load(path)
        except Exception:
            if RecurrentPPO is None:
                raise
            return RecurrentPPO.load(path)
    return PPO.load(path)


def is_recurrent(model) -> bool:
    return RecurrentPPO is not None and isinstance(model, RecurrentPPO)


def predict_action(model, obs, lstm_state, episode_start, deterministic):
    if is_recurrent(model):
        action, lstm_state = model.predict(
            obs, state=lstm_state, episode_start=episode_start, deterministic=deterministic
        )
        return action, lstm_state
    action, _ = model.predict(obs, deterministic=deterministic)
    return action, None


def evaluate_map(model, map_path: str, n_episodes: int, deterministic: bool = True):
    env = Maze3DEnv(map_path=map_path, randomize=False)
    steps_list, success_list, last_visits = [], [], None
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        while not done:
            action, lstm_state = predict_action(model, obs, lstm_state, episode_start, deterministic)
            episode_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        steps_list.append(env.steps)
        success_list.append(bool(info.get("is_success", False)))
        last_visits = env.visits.copy()
    return {
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "success_rate": float(np.mean(success_list)),
        "visits": last_visits,
        "raw_steps": steps_list,
    }


def plot_heatmap(visits: np.ndarray, grid: np.ndarray, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="binary", origin="upper", interpolation="nearest")
    if visits.sum() > 0:
        ax.imshow(np.log1p(visits), cmap="hot", alpha=0.7, origin="upper")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_trajectory_gif(model, map_path: str, out_path: str, max_steps: int = 200):
    env = Maze3DEnv(map_path=map_path, render_mode="rgb_array", randomize=False)
    obs, _ = env.reset(seed=0)
    frames = [env.render()]
    done = False
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)
    while not done and len(frames) < max_steps:
        action, lstm_state = predict_action(model, obs, lstm_state, episode_start, True)
        episode_start = np.zeros((1,), dtype=bool)
        obs, _, term, trunc, _ = env.step(action)
        frames.append(env.render())
        done = term or trunc
    imageio.mimsave(out_path, frames, fps=15)


def render_before_after(trained_model, map_path: str, out_path: str, max_steps: int = 200):
    """Side-by-side: random policy vs trained policy on the same map."""

    def rollout(model_or_none):
        env = Maze3DEnv(map_path=map_path, render_mode="rgb_array", randomize=False)
        obs, _ = env.reset(seed=0)
        frames = [env.render()]
        done = False
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        while not done and len(frames) < max_steps:
            if model_or_none is None:
                action = env.action_space.sample()
            else:
                action, lstm_state = predict_action(model_or_none, obs, lstm_state, episode_start, True)
                episode_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            done = term or trunc
        return frames

    before = rollout(None)
    after = rollout(trained_model)
    n = max(len(before), len(after))
    before += [before[-1]] * (n - len(before))
    after += [after[-1]] * (n - len(after))
    combined = [np.concatenate([b, a], axis=1) for b, a in zip(before, after)]
    imageio.mimsave(out_path, combined, fps=15)


def render_3d_trajectory(model, map_path: str, out_path: str, max_steps: int = 200):
    """3D plot: x = col, y = row, z = step. Saved as GIF rotating around."""
    env = Maze3DEnv(map_path=map_path, randomize=False)
    obs, _ = env.reset(seed=0)
    done = False
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)
    while not done and env.steps < max_steps:
        action, lstm_state = predict_action(model, obs, lstm_state, episode_start, True)
        episode_start = np.zeros((1,), dtype=bool)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    traj = np.array(env.trajectory)
    z = np.arange(len(traj))

    frames = []
    for angle in range(0, 360, 12):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(traj[:, 1], traj[:, 0], z, "r-", linewidth=2)
        ax.scatter([traj[0, 1]], [traj[0, 0]], [0], c="blue", s=80, label="start")
        ax.scatter([env.goal[1]], [env.goal[0]], [z[-1]], c="lime", s=120, marker="*", label="goal")
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_zlabel("step")
        ax.view_init(elev=25, azim=angle)
        ax.set_title("3D trajectory (z = timestep)")
        ax.legend()
        fig.canvas.draw()
        # tostring_rgb() removed in matplotlib 3.8+; buffer_rgba is the portable path
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(img)
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=12)


def plot_learning_curve_from_monitor(monitor_glob: str, out_path: str):
    """Pull rewards from Monitor CSVs (works without TB)."""
    files = glob.glob(monitor_glob)
    if not files:
        print(f"[bench] no monitor CSVs at {monitor_glob}; skipping learning curve")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in files:
        try:
            data = np.genfromtxt(f, delimiter=",", skip_header=2, names=True)
            if data.size == 0:
                continue
            ax.plot(np.cumsum(data["l"]), data["r"], alpha=0.6, label=os.path.basename(f))
        except Exception as e:
            print(f"  skip {f}: {e}")
    ax.set_xlabel("env steps (cumulative)")
    ax.set_ylabel("episode reward")
    ax.set_title("Learning curve (per-env Monitor logs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="agents/agent.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--maps-dir", type=str, default="maps")
    parser.add_argument("--viz-dir", type=str, default="viz")
    parser.add_argument("--monitor-glob", type=str, default="agents/*.monitor.csv")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of using deterministic mode "
                             "(diagnostic — useful if deterministic gives 0% success "
                             "to check whether the policy collapsed or just has a "
                             "high-variance action distribution)")
    args = parser.parse_args()

    Path(args.viz_dir).mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)
    print(f"[bench] loaded {args.model} ({type(model).__name__})")

    train_maps = sorted(glob.glob(os.path.join(args.maps_dir, "train*.npy")))
    test_maps = sorted(glob.glob(os.path.join(args.maps_dir, "test*.npy")))

    rows = []
    train_means = []
    test_means = []

    for path in train_maps + test_maps:
        name = os.path.basename(path).replace(".npy", "")
        res = evaluate_map(model, path, n_episodes=args.episodes,
                           deterministic=not args.stochastic)
        split = "train" if name.startswith("train") else "test"
        rows.append((split, name, res["mean_steps"], res["std_steps"], res["success_rate"]))
        (train_means if split == "train" else test_means).append(res["mean_steps"])
        print(
            f"  [{split}] {name}: {res['mean_steps']:.1f}±{res['std_steps']:.1f} steps, "
            f"success={res['success_rate']:.0%}"
        )

        env_for_grid = Maze3DEnv(map_path=path)
        plot_heatmap(
            res["visits"],
            env_for_grid.grid,
            os.path.join(args.viz_dir, f"heatmap_{name}.png"),
            title=f"Visit heatmap — {name}",
        )

    # Aggregate metrics
    train_mean = float(np.mean(train_means)) if train_means else 0.0
    test_mean = float(np.mean(test_means)) if test_means else 0.0
    gap = (test_mean - train_mean) / max(train_mean, 1e-6) * 100

    table_path = os.path.join(args.viz_dir, "results_table.txt")
    with open(table_path, "w") as f:
        f.write("split  map         mean±std steps    success\n")
        f.write("-" * 60 + "\n")
        for split, name, m, s, sr in rows:
            f.write(f"{split:<6} {name:<11} {m:>6.1f} ± {s:>5.1f}      {sr:>5.0%}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Train mean: {train_mean:.1f}\n")
        f.write(f"Test  mean: {test_mean:.1f}\n")
        f.write(f"Gen gap:    {gap:+.1f}%   (target: <40%)\n")
    print("\n" + open(table_path).read())

    # Visuals
    if test_maps:
        ref = test_maps[0]
        ref_name = os.path.basename(ref).replace(".npy", "")
        print(f"[bench] rendering trajectory + before/after on {ref_name}")
        render_trajectory_gif(model, ref, os.path.join(args.viz_dir, f"trajectory_{ref_name}.gif"))
        render_before_after(model, ref, os.path.join(args.viz_dir, "before_after.gif"))
        render_3d_trajectory(model, ref, os.path.join(args.viz_dir, "trajectory_3d.gif"))

    plot_learning_curve_from_monitor(
        args.monitor_glob,
        os.path.join(args.viz_dir, "learning_curve.png"),
    )

    print(f"\n[bench] all artifacts saved to {args.viz_dir}/")


if __name__ == "__main__":
    main()
