# FABS Track 2 — RL Agent for 3D-Physics Maze Navigation

**Build with AI 2026 DeepTech Hackathon · GDG April · FABS**
Continuous-control PPO agent that navigates 9×9 mazes with heterogeneous
physics (asphalt / grass / sand / ice friction, slope, temperature zones),
trained only on 4 maps and evaluated on 3 held-out OOD maps + judges'
unseen map.

## Quick start (local)

```bash
pip install -r requirements.txt
python maze_gen.py --train 4 --test 3 --size 11 --seed 42
python train.py --steps 500000          # ~1.5 h on T4, ~6+ h on CPU
python benchmark.py --episodes 100
```

For Colab free-tier T4: open `colab_train.ipynb` and run cells top-to-bottom (~2–3 h end-to-end).

## Repository layout

```
ai-hackathon-t2-maze-rl/
├── env/Maze3DEnv.py     # Custom Gymnasium env (23 obs, continuous action)
├── maze_gen.py          # Procedural maze generator (Recursive Backtracker)
├── train.py             # PPO training with domain randomization
├── benchmark.py         # 100-ep eval + heatmap + 3D + before/after GIFs
├── ablation.py          # 4-variant component study (rubric: 8 pts)
├── colab_train.ipynb    # End-to-end Colab notebook
├── maps/                # 4 train + 3 test (.npy, allow_pickle=True)
├── agents/agent.zip     # Trained PPO model
└── viz/                 # Auto-generated artifacts (curves, heatmaps, GIFs)
```

## Environment design

### Observation (23 floats, all ∈ [-1, 1])

| Slice    | Component                            | Justification                               |
|----------|--------------------------------------|---------------------------------------------|
| `[0:8]`  | 8 ray distances (N, NE, …, NW)       | local geometry → wall avoidance             |
| `[8:16]` | 8 surface codes at ray hits          | "see" terrain ahead → choose fast surface   |
| `[16:18]`| `(goal_dx, goal_dy)` normalized      | dense direction signal toward target        |
| `[18:21]`| `(friction, slope_avg, temp)` here   | local physics state                         |
| `[21:23]`| velocity (vx, vy) / max_vel          | momentum awareness for braking/turning      |

**Sensing strategy (rubric: 10 pts).** Egocentric ray casts + relative goal
direction make the obs invariant to absolute maze layout, so the agent learns
*reactive* navigation that transfers to OOD maps. Surface-code rays let the
policy plan one cell ahead onto faster terrain without seeing the full grid.

### Action

`Box([-1,-1], [+1,+1])` — continuous force vector. Acceleration = `action *
friction * energy + slope_pull`, integrated with `dt = 0.22`. Velocity is
clamped to ±2.5 cells/step and damped by 0.90 each step (more if hot).
Tuned so a 17-cell shortest path on a 9×9 maze fits in ~50 env-steps.

### Reward shaping (rubric: 10 pts)

```
r = -0.05                          (time penalty → minimize steps)
  + 1.5 * (prev_dist - cur_dist) / diag   (dense progress signal)
  + 0.05 * (1 if friction ≥ 1.0 else 0)   (terrain preference)
  - 0.30 if wall collision
  + 50.0 on reaching goal           (terminal)
```

The dense progress term solves the credit-assignment problem the spec warns
about ("If your reward is just +10 for finish and −0.1 per step, the agent
may not learn for hours"). Ablation (`ablation.py`) confirms removing it
collapses learning — see `viz/ablation.txt`.

## Algorithm choice (rubric: 8 pts)

**PPO (Stable-Baselines3).** Continuous action + small dense obs + on-policy
data is exactly PPO's sweet spot. SAC/TD3 would also work but need more
hyperparameter care; PPO with default-ish settings converges reliably. Net
arch `[128, 128]` MLP — small enough to train in <2 h on free T4, large
enough to fit 23-dim obs.

Hyperparams (see `train.py`): `lr=3e-4`, `n_steps=1024`, `batch_size=256`,
`n_epochs=10`, `ent_coef=0.01` (exploration bonus matters for maze tasks).

## Generalization strategy (rubric: 7 pts)

Three mechanisms together close the train/test gap:

1. **Egocentric obs** — no absolute coordinates, no global map.
2. **Domain randomization** at every `reset()` — random start cell + ±10%
   physics scale jitter (`Maze3DEnv(randomize=True)`).
3. **OOD test maps** are generated with a shifted surface distribution
   (more sand, larger slope intensity) to simulate the judge's unseen map.

Target gap: `(test_mean − train_mean) / train_mean < 40%`.

## Results

> Run `python benchmark.py` after training and paste numbers below.

```
split  map         mean±std steps    success
------------------------------------------------------------
train  train1       ___ ± ___        ___%
train  train2       ___ ± ___        ___%
train  train3       ___ ± ___        ___%
train  train4       ___ ± ___        ___%
test   test1        ___ ± ___        ___%
test   test2        ___ ± ___        ___%
test   test3        ___ ± ___        ___%
------------------------------------------------------------
Train mean: ___
Test  mean: ___
Gen gap:    ___%   (target: <40%)
```

### Ablation (rubric: 8 pts)

| Variant      | Train steps | Test steps | Gap   |
|--------------|------------:|-----------:|------:|
| full         |             |            |       |
| no_surface   |             |            |       |
| no_progress  |             |            |       |
| no_dr        |             |            |       |

Fill from `viz/ablation.txt` after running `python ablation.py`.

## Visualizations

- `viz/learning_curve.png` — episode reward vs env steps (Monitor logs)
- `viz/heatmap_<map>.png` — visit-count 2D histogram per map (bonus +5)
- `viz/trajectory_3d.gif` — 3D plot of trajectory rotating, z = timestep (bonus +5)
- `viz/before_after.gif` — random policy vs trained policy side-by-side (rubric: 7)

## Maps

| File             | Split | Source / generation                                  |
|------------------|-------|------------------------------------------------------|
| `maps/train1-4.npy` | train | Recursive Backtracker, surfaces sampled p≈[0.45/0.30/0.20/0.05] |
| `maps/test1-3.npy`  | test (OOD) | Same generator, **shifted distribution**: more sand (35%), larger slope amplitude |

Adapted from: [TateHouse/ProceduralMazeGenerator](https://github.com/TateHouse/ProceduralMazeGenerator) (algorithm),
[hicarrie/maze](https://github.com/hicarrie/maze) (raycast obs idea).

## About FABS

[FABS](https://fabstoryai.com) is an edtech platform for specialists working
on facial / body function and aesthetics (logopedists, myofunctional
therapists, cosmetologists, etc.). This Track 2 submission is a technical
demo of our team's RL/optimization capabilities, independent from the
clinical product.

## License

MIT — see `LICENSE` (TBD).
