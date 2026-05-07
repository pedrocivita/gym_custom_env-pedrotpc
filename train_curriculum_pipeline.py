#
# Curriculum learning pipeline for the v3.6 CPP agent.
#
# Trains on 5x5 from scratch, then transfers the weights to 10x10, then to
# 20x20. The v3.6 observation is minimal — agent (7,) + neighbors (5x5) —
# so the policy + extractor have identical parameter shapes across grid
# sizes (~110k params each); weights transfer directly between stages.
#
# Stage budgets follow the colleague Matheus's recipe ("treina em 5x5,
# 10x10, e um pouco em 20x20") which reached 100% full coverage:
# spend most compute on 5x5 and 10x10 to develop a robust coverage skill,
# then a smaller transfer pass on 20x20.
#
# Usage:
#   python train_curriculum_pipeline.py            # full 5x5 -> 10x10 -> 20x20
#   python train_curriculum_pipeline.py --dry-run  # print plan + ETA only
#   python train_curriculum_pipeline.py --skip-test
#

import argparse
import sys
import time
from datetime import datetime, timedelta

from train_grid_world_cpp import register_env, train_mode, curriculum_mode, test_mode


# (size, obstacles, max_steps, total_timesteps, fps_total, mode, transfer kwargs)
STAGES = [
    {
        "size": 5, "obstacles": 3, "max_steps": 200,
        "total_timesteps": 1_000_000,
        "fps_total": 1500,
        "mode": "scratch",
    },
    {
        "size": 10, "obstacles": 12, "max_steps": 600,
        "total_timesteps": 3_000_000,
        "fps_total": 800,
        "mode": "transfer",
        "lr": 1e-4,
        "ent_coef": 0.05,
    },
    {
        "size": 20, "obstacles": 48, "max_steps": 2000,
        "total_timesteps": 1_500_000,
        "fps_total": 500,
        "mode": "transfer",
        "lr": 5e-5,
        "ent_coef": 0.05,
    },
]


def fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def print_plan():
    print("=" * 60)
    print("v3.6 Curriculum Pipeline (5x5 -> 10x10 -> 20x20)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    cumulative = 0.0
    for s in STAGES:
        eta = s["total_timesteps"] / s["fps_total"]
        cumulative += eta
        mode = s["mode"]
        extra = f" lr={s.get('lr','-')} ent={s.get('ent_coef','-')}" if mode == "transfer" else ""
        print(
            f"  {s['size']:>2}x{s['size']:<2} | obs={s['obstacles']:<3} "
            f"max_steps={s['max_steps']:<5} "
            f"timesteps={s['total_timesteps']:>10,} "
            f"mode={mode:<8}{extra} "
            f"~ETA={fmt_eta(eta)}"
        )
    print(f"  Total estimated wall-time: {fmt_eta(cumulative)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without training")
    parser.add_argument("--skip-test", action="store_true",
                        help="Do not run the 100-episode test after each stage")
    args = parser.parse_args()

    print_plan()
    if args.dry_run:
        return

    register_env()
    overall_start = time.time()
    results = {}
    last_model_path = None

    for stage in STAGES:
        s = stage["size"]
        stage_start = time.time()
        print(f"\n>>> Stage {s}x{s} starting at {datetime.now().strftime('%H:%M:%S')}")
        try:
            if stage["mode"] == "scratch":
                model_path = train_mode(
                    dim=s,
                    obstacles=stage["obstacles"],
                    max_steps=stage["max_steps"],
                    total_timesteps=stage["total_timesteps"],
                )
            else:
                if last_model_path is None:
                    print(f"!!! No prior model to transfer from for stage {s}x{s} -- abort")
                    break
                model_path = curriculum_mode(
                    dim=s,
                    obstacles=stage["obstacles"],
                    max_steps=stage["max_steps"],
                    total_timesteps=stage["total_timesteps"],
                    base_model_path=last_model_path,
                    learning_rate=stage["lr"],
                    ent_coef=stage["ent_coef"],
                )
        except Exception as e:
            print(f"!!! Stage {s}x{s} failed: {e}")
            print("Continuing to next stage to avoid losing the rest of the run.")
            continue

        last_model_path = model_path
        elapsed = time.time() - stage_start
        print(f">>> Stage {s}x{s} finished in {fmt_eta(elapsed)} -> {model_path}")

        if not args.skip_test:
            try:
                metrics = test_mode(s, stage["obstacles"], model_path)
                results[s] = metrics
            except Exception as e:
                print(f"!!! Test for {s}x{s} failed: {e}")

    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"Pipeline finished in {fmt_eta(total_elapsed)}")
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
