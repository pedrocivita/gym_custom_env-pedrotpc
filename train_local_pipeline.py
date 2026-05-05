#
# Sequential overnight training pipeline for the v3 CPP agent.
#
# Trains three independent MaskablePPO models (5x5, 10x10, 20x20) using the
# train_mode entry point, in series. Designed to be launched once before
# leaving the machine for the night.
#
# Usage:
#   python train_local_pipeline.py            # full pipeline (5, 10, 20)
#   python train_local_pipeline.py 5 10       # only the listed sizes
#   python train_local_pipeline.py --dry-run  # print plan + ETA, no training
#

import argparse
import sys
import time
from datetime import datetime, timedelta

from train_grid_world_cpp import register_env, train_mode, test_mode


# (size, obstacles, max_steps, total_timesteps, est_fps_per_env)
# est_fps_per_env is rough; used to print an ETA before training starts.
STAGES = {
    5:  dict(obstacles=3,  max_steps=100,  total_timesteps=300_000,   fps_total=600),
    10: dict(obstacles=12, max_steps=400,  total_timesteps=1_200_000, fps_total=350),
    20: dict(obstacles=48, max_steps=1500, total_timesteps=2_000_000, fps_total=160),
}


def fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def print_plan(sizes):
    print("=" * 60)
    print("v3 CPP Training Pipeline — overnight schedule")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    cumulative_eta = 0.0
    for s in sizes:
        cfg = STAGES[s]
        eta = cfg["total_timesteps"] / cfg["fps_total"]
        cumulative_eta += eta
        print(
            f"  {s:>2}x{s:<2} | obstacles={cfg['obstacles']:<3} "
            f"max_steps={cfg['max_steps']:<5} "
            f"timesteps={cfg['total_timesteps']:>10,} "
            f"~ETA={fmt_eta(eta)}"
        )
    print(f"  Total estimated wall-time: {fmt_eta(cumulative_eta)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="*", type=int, default=list(STAGES.keys()),
                        help="Grid sizes to train (default: 5 10 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without training")
    parser.add_argument("--skip-test", action="store_true",
                        help="Do not run the 100-episode test after each stage")
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
            print("Continuing to next stage to avoid losing the rest of the night.")
            continue

        elapsed = time.time() - stage_start
        print(f">>> Stage {s}x{s} finished in {fmt_eta(elapsed)} → {model_path}")

        if not args.skip_test:
            try:
                metrics = test_mode(s, cfg["obstacles"], model_path)
                results[s] = metrics
            except Exception as e:
                print(f"!!! Test for {s}x{s} failed: {e}")

    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"Pipeline finished in {fmt_eta(total_elapsed)}")
    print("=" * 60)
    if results:
        for s, m in results.items():
            print(
                f"  {s}x{s}: full_coverage={m['full_coverage_rate']:.1f}% "
                f"avg_coverage={m['avg_coverage']:.1f}% "
                f"avg_steps={m['avg_steps']:.1f}"
            )


if __name__ == "__main__":
    main()
