# Handoff: APS Coverage Path Planning — Session Context

> **Date**: 2026-05-04
> **Author**: Claude (Opus 4.6) assisting Pedro Civita
> **Repo**: `github.com/pedrocivita/gym_custom_env-pedrotpc`
> **Deadline**: 2026-05-08
> **Goal**: Grade 10 (5x5 + 10x10 ~100% coverage) or 11 (+ 20x20 bonus)

---

## 1. The Assignment

The professor's RL course (Insper, 10th semester) provides a forked Gymnasium repo with a GridWorld environment for Coverage Path Planning (CPP). The agent must visit every free cell in grids of varying sizes while avoiding obstacles. The baseline agent (professor's code) achieves only ~75-81% on 5x5 and ~59-70% on 10x10. Students must improve it to near-100%.

**Grading**:
- Grade 10: ~100% coverage on 5x5 AND 10x10
- Grade 11 (bonus): also ~100% on 20x20

**Deliverables**: Modified code, trained models, `report_cpp.md` with methodology + results, updated `README.md`.

---

## 2. What Was Wrong With the Baseline

The professor's original agent had several fundamental limitations:

1. **No memory** — Used `MultiInputPolicy` (pure MLP), so the agent is purely reactive. It can't remember where it's been beyond what the observation tells it. This is fatal for CPP where trajectory history matters.

2. **Tiny field of view** — Only a 3x3 neighbor matrix (9 cells). On a 10x10 grid, this is <10% of the grid. The agent is essentially blind to the larger structure.

3. **No directional guidance** — The observation was just `[norm_x, norm_y, coverage_ratio]` (3 floats) + 3x3 neighbors. The agent had no signal about WHERE unexplored regions are — it could only see immediate neighbors.

4. **Insufficient training** — 1M timesteps with default hyperparameters, single environment (no parallelism).

5. **No curriculum** — Training directly on larger grids is extremely hard from scratch with limited compute.

6. **Script bugs** — The `train_grid_world_cpp.py` had argparse issues (test/run modes crashed with IndexError) and the curriculum mode did a double `model.learn()` call.

---

## 3. Strategy: Six Pillars of Improvement

We designed a comprehensive strategy with 6 changes, ordered by impact:

### 3.1 RecurrentPPO (LSTM) — HIGH IMPACT
Switched from `PPO` with `MultiInputPolicy` to `RecurrentPPO` from `sb3-contrib` with `MultiInputLstmPolicy`. This gives the agent implicit trajectory memory through LSTM hidden states — it can remember where it's been even without explicit encoding in the observation.

**Technical detail**: `lstm_hidden_size=128`, `n_lstm_layers=1`, `shared_lstm=False`, `enable_critic_lstm=True` (separate LSTM for actor and critic).

### 3.2 Expanded Neighbor View (5x5) — HIGH IMPACT
Expanded the neighbor matrix from 3x3 to 5x5, giving the agent 25 cells of local information instead of 9. This is ~2.8x more spatial context.

**Implementation**: Changed `set_neighbors()` from a Python double-loop with per-cell `any(np.array_equal(...))` to a numpy padded-array slicing approach. The padded grid is `(size+4) x (size+4)`, pre-filled with 1 (wall), then the 5x5 window is a simple slice. This was both a feature improvement AND a performance optimization.

### 3.3 Directional Exploration Signals — MEDIUM IMPACT
Added 4 floats to the "agent" observation vector: `[unvisited_ratio_right, unvisited_ratio_up, unvisited_ratio_left, unvisited_ratio_down]`. Each value is `(unvisited_free_cells_in_direction / total_free_cells_in_direction)`.

This tells the agent "there's 80% unexplored to the right, 10% to the left" — a legitimate signal under partial observability (the agent knows where it HAS been, just not the full map).

**Implementation**: Vectorized with numpy meshgrids + boolean masks. Pre-built `self._xs, self._ys` coordinate arrays in `__init__`, then each direction is a single boolean mask operation.

### 3.4 Custom CNN Feature Extractor — MEDIUM IMPACT
Created `utils/feature_extractor.py` with `CustomCombinedExtractor(BaseFeaturesExtractor)`:
- **neighbor_cnn**: Conv2d(1→16, k=3, pad=1) → ReLU → Conv2d(16→32, k=3, pad=1) → ReLU → Flatten
- **agent_mlp**: Linear(7→64) → ReLU
- **combine**: Linear(cnn_out + 64 → 128) → ReLU

This processes the 5x5 spatial grid through convolutions (detecting patterns like walls, corridors, visited clusters) and combines with the agent vector.

### 3.5 Curriculum Learning — HIGH IMPACT (but see caveats below)
Pipeline: Train on 5x5 → Transfer to 10x10 → Transfer to 20x20. The idea is that small-grid skills (obstacle avoidance, systematic coverage) transfer to larger grids.

**CAVEAT**: The 5x5→10x10 curriculum FAILED in v1 (see Section 5). In v2 we train 10x10 from scratch, then do curriculum only for 10x10→20x20.

### 3.6 Reward Shaping — LOW IMPACT
Minor tweaks to the reward function:
- **Progressive revisit penalty**: `-0.3 * (1.0 + consecutive_revisits * 0.1)`, capped at -1.0
- **Scaled completion bonus**: `+10.0 * (size / 5.0)` (so 20x20 gets +40 instead of +10)
- **Consecutive revisit tracker**: `self.consecutive_revisits` resets on new cell

---

## 4. Implementation Details

### Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `gymnasium_env/grid_world_cpp.py` | Modified | Expanded obs (5x5 neighbors, 7-float agent), vectorized internals, reward shaping |
| `train_grid_world_cpp.py` | Rewritten | RecurrentPPO, LSTM predict loop, fixed argparse/curriculum bugs |
| `utils/__init__.py` | Created | Empty, makes utils a package |
| `utils/feature_extractor.py` | Created | CustomCombinedExtractor (CNN + MLP) |
| `train_curriculum_pipeline.py` | Created | Automated 3-stage pipeline for local training |
| `train_colab.ipynb` | Created | v1 Colab notebook (has issues, see below) |
| `train_colab_v2.ipynb` | Created | v2 Colab notebook (optimized, fixes v1 issues) |
| `report_cpp.md` | Created | Report template (results still TBD) |
| `README.md` | Updated | Added "Improved CPP Agent" section |
| `requirements.txt` | Updated | Added sb3-contrib, switched pygame→pygame-ce, added torch |

### Performance Optimizations in the Environment

The environment code was heavily optimized for throughput:

1. **Obstacle lookup**: O(n) `any(np.array_equal(...))` → O(1) `set` lookup via `self._obstacle_set`
2. **Obstacle grid**: Pre-built `self._obstacle_grid` (boolean numpy array) for vectorized operations
3. **Neighbor computation**: Python double-loop with per-cell checks → numpy padded array slicing
4. **Directional signals**: Fully vectorized with pre-built meshgrid coordinate arrays
5. **Obstacle placement in reset()**: O(n²) → O(1) with incremental set building

Raw env benchmark results: ~8360 FPS (5x5), ~7526 FPS (10x10) — ~200-400x faster than naive implementation.

### Hyperparameters Used

```python
# 5x5 (both v1 and v2)
RecurrentPPO(
    "MultiInputLstmPolicy", env,
    learning_rate=3e-4, n_steps=256, batch_size=128, n_epochs=4,
    gamma=0.995, gae_lambda=0.95, ent_coef=0.05,
    clip_range=0.2, max_grad_norm=0.5,
)

# 10x10 v2 (from scratch)
# Same as 5x5 but ent_coef=0.08, max_steps=600

# 20x20 v2 (curriculum from 10x10)
# learning_rate=1e-4, ent_coef=0.1, max_steps=2000
```

---

## 5. What Happened During Training (Chronological)

### 5.1 Local Training Attempt (Failed — Too Slow)
- Pedro's machine: Lenovo Yoga Book 9i, Intel Ultra 7 (no NVIDIA GPU)
- Training speed: ~42-46 FPS on CPU with LSTM
- Estimated time for full pipeline: 30+ hours
- **Decision**: Pivot to Google Colab with T4 GPU

### 5.2 Colab v1 — 5x5 Training (SUCCESS)
- 2M timesteps, T4 GPU, ~205-440 FPS
- Training curve: ep_rew went from -77 → +21.2, ep_len from 147 → 32
- **The 5x5 agent learned to efficiently cover the grid**
- Took approximately 80 minutes

### 5.3 Colab v1 — 10x10 Curriculum (FAILED)
This is the critical failure point. The 5x5 model was loaded and trained on 10x10 with 1.5M timesteps. What happened:

- **Iteration 1**: ep_rew=+25.9, ep_len=39 (inherited from 5x5 — misleading because these are leftover 5x5 stats)
- **By iteration 50**: ep_rew=-181, ep_len=490
- **By iteration 170**: ep_rew=-208, ep_len=500 (max_steps — every episode truncating)
- **Entropy collapsing**: entropy_loss went from -0.7 to -0.64
- **Policy frozen**: clip_fraction dropped to 0.024 (policy barely updating)

**Root cause analysis**:
1. The 5x5 agent learned a very specific policy for 22 free cells. When placed in 88 free cells (10x10), these strategies don't generalize.
2. The learned policy deterministically follows a 5x5-optimal path, which in 10x10 just means wandering in a small area.
3. With `ent_coef=0.05` and a strong prior from 5x5, the agent doesn't explore enough to discover 10x10 strategies.
4. 1.5M timesteps is far too few for the agent to "unlearn" the 5x5 policy and find a new one.
5. `max_steps=500` may be too low for 10x10 (need ~88 new cells + movement overhead).

### 5.4 Colab v2 — The Fix (PENDING)
Created `train_colab_v2.ipynb` with these changes:
- **10x10 from scratch** (no curriculum from 5x5 — it poisoned the policy)
- **5M timesteps** for 10x10 (3.3x more than v1)
- **`ent_coef=0.08`** for 10x10 (60% more exploration)
- **`max_steps=600`** for 10x10 (more room to complete coverage)
- **`n_envs=16`** (double v1's 8, better GPU utilization)
- **Curriculum only 10x10→20x20** (same observation space, more similar scales)
- **`learning_rate=1e-4`** and **`ent_coef=0.1`** for 20x20 curriculum
- **Checkpoints every 500k steps** (never lose progress if Colab disconnects)
- **Google Drive auto-save** for models and results
- **CUDA optimizations**: `cudnn.benchmark=True`, `tf32` enabled

**Estimated runtime**: ~6-7 hours total (designed to run overnight).

---

## 6. Current State (as of 2026-05-04 evening)

### Git State
- **Branch**: `main`
- **Latest commit**: `cf7d04e` — "Add optimized v2 Colab notebook for overnight training"
- **All changes pushed** to `github.com/pedrocivita/gym_custom_env-pedrotpc`

### Commits Made This Session
1. `ceb022a` — Main implementation (env, training script, extractor, pipeline, report, README)
2. `e0fcdd3` — Fix torch.cuda `total_mem` → `total_global_mem`
3. `b72bee1` — Reduce Colab v1 timesteps to fit free tier
4. `cf7d04e` — Add v2 Colab notebook

### Training Status
- **v1 notebook**: May still be running on Colab. 5x5 model is good; 10x10 and 20x20 will be bad.
- **v2 notebook**: Ready to run. Not yet started.

### Files with TBD Content
- **`report_cpp.md`**: Results tables have `TBD` placeholders. Needs actual numbers from training.
- The report structure is complete — just needs data filled in.

---

## 7. What the Next Claude Instance Needs To Do

### Step 1: Check Training Results
When Pedro comes back with results from v2 Colab training (or v1 if he used those), the next step is to analyze the test outputs:
- 5x5: Should be >95% full coverage rate
- 10x10: Target >80%, ideally >90%
- 20x20: Any improvement over baseline is good for bonus

### Step 2: Fill In report_cpp.md
The report at `report_cpp.md` has a complete structure but needs:
- Actual coverage rates, avg coverage %, avg steps for each grid size
- The results tables (currently TBD)
- Possibly add a section about the v1→v2 iteration (curriculum failure analysis)

**Important**: The report should tell the story of the development process — the curriculum failure and recovery is actually great content for the report. It shows understanding of RL dynamics.

### Step 3: If Results Are Poor
If v2 still doesn't achieve good 10x10 results:
- **Try more timesteps** (8-10M for 10x10)
- **Try different approach**: Train with `max_steps=400` first (easier to complete), then increase
- **Try intermediate curriculum**: 5x5 → 7x7 → 10x10 (smaller jumps)
- **Consider reward changes**: Increase new-cell reward to +2.0, reduce step penalty

### Step 4: Final Commit
- Update `report_cpp.md` with real results
- Commit and push
- Ensure no TBD placeholders remain

---

## 8. Technical Gotchas and Lessons Learned

### Python 3.14 + pygame
Pedro's local Python 3.14 broke `pygame 2.6.1` build (`ModuleNotFoundError: No module named 'distutils.msvccompiler'`). Fixed by using `pygame-ce` (community edition) which supports 3.14.

### PowerShell + Python f-strings
Running inline Python with dict keys (`obs["agent"]`) inside PowerShell causes parsing issues. Solution: write to a `.py` file and run it, or use escaped syntax.

### LSTM Makes Training SLOW
RecurrentPPO with LSTM is ~10x slower than standard PPO per timestep because of sequential backpropagation through time. This is why GPU is essential — LSTM backprop benefits heavily from GPU acceleration.

On CPU: ~42-46 FPS. On T4 GPU: ~200-440 FPS (4-10x speedup).

### Curriculum Learning is Not Always Better
The 5x5→10x10 curriculum poisoned the agent because:
- The scale difference is too large (22 cells → 88 cells, 4x)
- The 5x5 policy is too deterministic (low entropy after convergence)
- The learning rate was too high for fine-tuning (3e-4)

Lesson: Curriculum works best when the gap between stages is small AND the curriculum stage uses lower LR + higher entropy to allow policy adaptation.

### Observation Space Must Be Identical Across Curriculum Stages
The 5x5 neighbor view works the same on any grid size (it's always a 5x5 window centered on the agent). The 7-float agent vector is also size-invariant (normalized coordinates, ratios). This is critical — if the observation space changed between stages, the model weights wouldn't transfer.

### Checkpoint Frequency
v1 had no checkpoints. If Colab disconnected, everything was lost. v2 saves every 500k steps AND copies to Google Drive. Always do this for long training runs.

---

## 9. Observation Space Reference

```
observation_space = Dict({
    "agent": Box(shape=(7,), low=0, high=1, dtype=float32)
        [0] norm_x = agent_x / (size - 1)
        [1] norm_y = agent_y / (size - 1)
        [2] coverage_ratio = visited / total_free_cells
        [3] unvisited_ratio_right  (cells with x > agent_x)
        [4] unvisited_ratio_up     (cells with y < agent_y)
        [5] unvisited_ratio_left   (cells with x < agent_x)
        [6] unvisited_ratio_down   (cells with y > agent_y)

    "neighbors": Box(shape=(5,5), low=0, high=2, dtype=float32)
        0 = free, unvisited
        1 = obstacle or wall (out of bounds)
        2 = visited
        Center (2,2) is always the agent's position
})

action_space = Discrete(4)
    0 = right (+x)
    1 = up (-y)
    2 = left (-x)
    3 = down (+y)
```

---

## 10. Reward Function Reference

| Condition | Reward |
|-----------|--------|
| Per-step penalty (always) | -0.1 |
| Visit new cell | +1.0 |
| Revisit cell | -0.3 × (1 + consecutive_revisits × 0.1), capped at -1.0 |
| Hit wall/obstacle (stay in place) | -0.5 |
| Full coverage achieved | +10.0 × (size / 5.0) |
| Truncation (max_steps reached) | -5.0 |

---

## 11. User Preferences (for next Claude instance)

- **Language**: Code/commits in English, explanations in Portuguese (pt-BR with correct accents)
- **No Co-Authored-By**: NEVER add Claude co-author trailers to git commits (saved in memory)
- **Shell**: PowerShell 5.1 on Windows 11. No bash-only syntax.
- **Concise**: Pedro prefers direct answers, not long explanations. He's a senior dev.
- **Mobile**: Pedro often works from phone — keep messages scannable.
- **Git**: Always commit + push after changes. Never leave uncommitted work.
