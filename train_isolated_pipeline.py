#
# Isolated training pipeline for v3.5 — each grid size trained from scratch
# without curriculum / weight transfer. The v3.5 observation includes a
# global visited_map whose shape depends on grid size, so weights cannot
# transfer between sizes — but each size can converge on its own with
# enough timesteps.
#
# Usage:
#   python train_isolated_pipeline.py
#   python train_isolated_pipeline.py --dry-run
#   python train_isolated_pipeline.py 10 20      # only retrain those sizes
#   python train_isolated_pipeline.py --skip-test
#

import argparse
import sys
import time
from datetime import datetime, timedelta

from train_grid_world_cpp import register_env, train_mode, test_mode


# (size, obstacles, max_steps, total_timesteps, fps_total)
STAGES = {
    5:  dict(obstacles=3,  max_steps=200,  total_timesteps=1_000_000, fps_total=1200),
    10: dict(obstacles=12, max_steps=600,  total_timesteps=3_000_000, fps_total=600),
    20: dict(obstacles=48, max_steps=2000, total_timesteps=5_000_000, fps_total=280),
}


def fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def print_plan(sizes):
    print("=" * 60)
    print("v3.5 Isolated Training Pipeline (each size from scratch)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    cumulative = 0.0
    for s in sizes:
        cfg = STAGES[s]
        eta = cfg["total_timesteps"] / cfg["fps_total"]
        cumulative += eta
        print(
            f"  {s:>2}x{s:<2} | obstacles={cfg['obstacles']:<3} "
            f"max_steps={cfg['max_steps']:<5} "
            f"timesteps={cfg['total_timesteps']:>10,} "
            f"~ETA={fmt_eta(eta)}"
        )
    print(f"  Total estimated wall-time: {fmt_eta(cumulative)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="*", type=int, default=list(STAGES.keys()))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip the 100-episode dual test after each stage")
    args = parser.parse_args()

    sizes = [s for s in args.sizes if s in STAGES]
    if not sizes:
        print(f"No valid sizes given. Choose from: {list(STAGES.keys())}")
        sys.exit(1)

    print_plan(sizes)
    if args.dry_run:
        return

    register_env()
    overall_start = time.time()
    results = {}

    for s in sizes:
        cfg = STAGES[s]
        stage_start = time.time()
        print(f"\n>>> Stage {s}x{s} starting at {datetime.now().strftime('%H:%M:%S')}")
        try:
            model_path = train_mode(
                dim=s,
                obstacles=cfg["obstacles"],
                max_steps=cfg["max_steps"],
                total_timesteps=cfg["total_timesteps"],
            )
        except Exception as e:
            print(f"!!! Stage {s}x{s} failed: {e}")
            print("Continuing to next stage to avoid losing the rest of the run.")
            continue

        elapsed = time.time() - stage_start
        print(f">>> Stage {s}x{s} finished in {fmt_eta(elapsed)} -> {model_path}")

        if not args.skip_test:
            try:
                metrics = test_mode(s, cfg["obstacles"], model_path)
                results[s] = metrics
            except Exception as e:
                print(f"!!! Test for {s}x{s} failed: {e}")

    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"Isolated pipeline finished in {fmt_eta(total_elapsed)}")
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
