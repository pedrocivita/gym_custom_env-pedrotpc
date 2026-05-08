#
# v3.6 polish v2 — aggressive entropy compression to push 10x10 stoch
# full coverage from 68% (after first polish with ent=0.01) toward >=90%.
#
# First polish (ent_coef=0.01, lr=5e-5, 1M tsteps) only dropped
# entropy_loss from -1.13 -> -1.03. The 5x5 success case has
# entropy_loss = -0.75. We need a more aggressive ent_coef to compress
# the action distribution further so the agent commits to closing the
# last few cells.
#
# Sequencing:
#   1) 10x10 polish v2 from the previous polished checkpoint
#      (ent_coef=0.005, lr=5e-5, 1.5M tsteps -> ~14 min)
#   2) 20x20 polish from the v2 10x10 model
#      (ent_coef=0.005, lr=2e-5, 1M tsteps -> ~15 min)
#
# Total: ~30 min on Pedro's CPU.
#

import argparse
import glob
import os
import time
from datetime import datetime, timedelta

from train_grid_world_cpp import register_env, curriculum_mode, test_mode


POLISH_V2_STAGES = [
    {
        "size": 10, "obstacles": 12, "max_steps": 600,
        "total_timesteps": 1_500_000,
        "lr": 5e-5,
        "ent_coef": 0.005,
        # Latest 10x10 polish output (the 68% one).
        "base_glob": "data/maskppo_cpp_10_12_600_*_curr.zip",
    },
    {
        "size": 20, "obstacles": 48, "max_steps": 2000,
        "total_timesteps": 1_000_000,
        "lr": 2e-5,
        "ent_coef": 0.005,
        # Will use freshly-polished v2 10x10 as base.
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
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--only-10", action="store_true",
                        help="Only polish 10x10 (skip 20x20)")
    args = parser.parse_args()

    print("=" * 60)
    print("v3.6 Polish v2 (ent=0.005, more aggressive entropy compression)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    register_env()
    overall_start = time.time()
    last_path: str | None = None
    results = {}

    for stage in POLISH_V2_STAGES:
        s = stage["size"]
        if args.only_10 and s != 10:
            continue

        if stage["base_glob"]:
            base = latest_model(stage["base_glob"])
        else:
            base = last_path

        if not base:
            print(f"!!! No base model for {s}x{s}; skipping")
            continue

        print(f"\n>>> Polish v2 {s}x{s} starting at {datetime.now().strftime('%H:%M:%S')}")
        print(f"    Base: {base}")
        print(f"    lr={stage['lr']} ent_coef={stage['ent_coef']} "
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
            print(f"!!! Polish v2 {s}x{s} failed: {e}")
            continue

        last_path = model_path
        elapsed = time.time() - stage_start
        print(f">>> Polish v2 {s}x{s} done in {fmt_eta(elapsed)} -> {model_path}")

        if not args.skip_test:
            try:
                metrics = test_mode(s, stage["obstacles"], model_path)
                results[s] = metrics
            except Exception as e:
                print(f"!!! Test {s}x{s} failed: {e}")

    print("\n" + "=" * 60)
    print(f"Polish v2 done in {fmt_eta(time.time()-overall_start)}")
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
