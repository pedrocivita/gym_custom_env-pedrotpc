#
# v3.6 polish pass — fine-tunes the curriculum-trained 10x10 and 20x20
# models with reduced entropy regularization (ent_coef=0.01 vs 0.05) and
# a small learning rate so the policy commits to closing the last few
# cells without destroying the coverage structure already learned.
#
# Diagnosis (post-pipeline):
#   5x5  stoch full=92%  avg=98.7%  -> already at target (ent_coef=0.01)
#   10x10 stoch full=53% avg=98.9% -> agent covers 98.9% but rarely
#                                     closes the last cell because the
#                                     policy is near-uniform (entropy_loss
#                                     ~ -1.13, max possible is -1.386 for
#                                     4 actions).
#   20x20 stoch full=0% avg=97.1%  -> same pattern, more pronounced.
#
# Hypothesis: ent_coef=0.05 was good for transfer (avoid catastrophic
# forgetting), but kept the policy too exploratory at convergence. A
# polish pass with ent_coef=0.01 should sharpen the action distribution
# enough to commit to the last few unvisited cells.
#
# Usage:
#   python polish_pipeline.py            # polish 10x10 then 20x20
#   python polish_pipeline.py --only-10  # only 10x10
#   python polish_pipeline.py --skip-test
#

import argparse
import glob
import os
import time
from datetime import datetime, timedelta

from train_grid_world_cpp import register_env, curriculum_mode, test_mode


POLISH_STAGES = [
    {
        "size": 10, "obstacles": 12, "max_steps": 600,
        "total_timesteps": 1_000_000,
        "lr": 5e-5,
        "ent_coef": 0.01,
        # Pick the latest curriculum 10x10 model (the one we just trained).
        "base_glob": "data/maskppo_cpp_10_12_600_*_curr.zip",
    },
    {
        "size": 20, "obstacles": 48, "max_steps": 2000,
        "total_timesteps": 1_000_000,
        "lr": 2e-5,
        "ent_coef": 0.01,
        # Will use the polished 10x10 output as base if available.
        "base_glob": None,
    },
]


def fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def latest_model(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip the dual-mode test after each polish stage")
    parser.add_argument("--only-10", action="store_true",
                        help="Only polish 10x10 (skip 20x20)")
    args = parser.parse_args()

    print("=" * 60)
    print("v3.6 Polish Pipeline (lower ent_coef -> commit to close)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    register_env()
    overall_start = time.time()
    last_path: str | None = None
    results = {}

    for stage in POLISH_STAGES:
        s = stage["size"]
        if args.only_10 and s != 10:
            continue

        # Resolve base model
        if stage["base_glob"]:
            base = latest_model(stage["base_glob"])
        else:
            base = last_path  # use freshly-polished previous stage

        if not base:
            print(f"!!! No base model found for {s}x{s} polish; skipping")
            continue

        print(f"\n>>> Polish {s}x{s} starting at {datetime.now().strftime('%H:%M:%S')}")
        print(f"    Base model: {base}")
        print(f"    Hyperparams: lr={stage['lr']} ent_coef={stage['ent_coef']} "
              f"timesteps={stage['total_timesteps']:,}")

        stage_start = time.time()
        try:
            model_path = curriculum_mode(
                dim=s,
                obstacles=stage["obstacles"],
                max_steps=stage["max_steps"],
                total_timesteps=stage["total_timesteps"],
                base_model_path=base,
                learning_rate=stage["lr"],
                ent_coef=stage["ent_coef"],
            )
        except Exception as e:
            print(f"!!! Polish {s}x{s} failed: {e}")
            continue

        last_path = model_path
        elapsed = time.time() - stage_start
        print(f">>> Polish {s}x{s} finished in {fmt_eta(elapsed)} -> {model_path}")

        if not args.skip_test:
            try:
                metrics = test_mode(s, stage["obstacles"], model_path)
                results[s] = metrics
            except Exception as e:
                print(f"!!! Test {s}x{s} failed: {e}")

    total = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"Polish pipeline finished in {fmt_eta(total)}")
    print("=" * 60)
    if results:
        for s, m in results.items():
            det = m.get("deterministic", {})
            sto = m.get("stochastic", {})
            print(
                f"  {s}x{s}: det full={det.get('full_coverage_rate', 0):.1f}%"
                f" avg={det.get('avg_coverage', 0):.1f}% | "
                f"stoch full={sto.get('full_coverage_rate', 0):.1f}%"
                f" avg={sto.get('avg_coverage', 0):.1f}%"
            )


if __name__ == "__main__":
    main()
