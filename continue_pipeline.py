#
# Continue-training pipeline for 10x10 and 20x20.
#
# Loads the latest existing models for each size and continues training
# from the same checkpoint with reduced learning rate and slightly higher
# entropy. Goal: take 10x10 from ~77% to >=90% full coverage (stochastic),
# and try to lift 20x20 from 1% toward something defensible.
#
# Usage:
#   python continue_pipeline.py
#   python continue_pipeline.py --dry-run
#   python continue_pipeline.py --skip-test
#

import argparse
import time
from datetime import datetime, timedelta

from train_grid_world_cpp import (
    register_env,
    curriculum_mode,
    test_mode,
    find_latest_model,
)


STAGES = [
    {
        "size": 10, "obstacles": 12, "max_steps": 600,
        "total_timesteps": 1_500_000,
        "fps_total": 700,
        "lr": 1e-4,       # match curriculum lr (5e-5 was too conservative)
        "ent_coef": 0.05, # bump entropy to break out of any deterministic loops
        "gamma": 0.997,   # lift discount so the long-horizon full-coverage
                          # bonus stays visible (0.99^600 ≈ 2e-3 was too small)
    },
    {
        "size": 20, "obstacles": 48, "max_steps": 2000,
        "total_timesteps": 4_000_000,  # +1M vs original; gamma helps but
                                       # 20x20 is the hardest stage
        "fps_total": 400,
        "lr": 1e-4,
        "ent_coef": 0.05,
        "gamma": 0.997,   # critical at this scale: 0.99^2000 ≈ 0 made the
                          # full-coverage bonus invisible to the policy
    },
]


def fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def print_plan():
    print("=" * 60)
    print("Continue-Training Pipeline (10x10 + 20x20 from latest)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    cumulative = 0.0
    for s in STAGES:
        eta = s["total_timesteps"] / s["fps_total"]
        cumulative += eta
        latest = find_latest_model(s["size"], s["obstacles"])
        print(
            f"  {s['size']:>2}x{s['size']:<2} | obs={s['obstacles']:<3} "
            f"max_steps={s['max_steps']:<5} "
            f"timesteps={s['total_timesteps']:>10,} "
            f"lr={s['lr']} ent={s['ent_coef']} gamma={s.get('gamma', 0.997)} "
            f"~ETA={fmt_eta(eta)}"
        )
        print(f"          base = {latest if latest else '(NONE FOUND)'}")
    print(f"  Total estimated wall-time: {fmt_eta(cumulative)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip the 100-episode dual test after each stage")
    args = parser.parse_args()

    print_plan()
    if args.dry_run:
        return

    register_env()
    overall_start = time.time()
    results = {}

    for stage in STAGES:
        s = stage["size"]
        base_path = find_latest_model(s, stage["obstacles"])
        if base_path is None:
            print(f"!!! No base model found for {s}x{s} -- skipping")
            continue

        stage_start = time.time()
        print(f"\n>>> Continue {s}x{s} starting at {datetime.now().strftime('%H:%M:%S')}")
        print(f"    base = {base_path}")

        try:
            model_path = curriculum_mode(
                dim=s,
                obstacles=stage["obstacles"],
                max_steps=stage["max_steps"],
                total_timesteps=stage["total_timesteps"],
                base_model_path=base_path,
                learning_rate=stage["lr"],
                ent_coef=stage["ent_coef"],
                gamma=stage.get("gamma", 0.997),
            )
        except Exception as e:
            print(f"!!! Continue stage {s}x{s} failed: {e}")
            continue

        elapsed = time.time() - stage_start
        print(f">>> Continue {s}x{s} finished in {fmt_eta(elapsed)} -> {model_path}")

        if not args.skip_test:
            try:
                metrics = test_mode(s, stage["obstacles"], model_path)
                results[s] = metrics
            except Exception as e:
                print(f"!!! Test for {s}x{s} failed: {e}")

    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"Continue-pipeline finished in {fmt_eta(total_elapsed)}")
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
