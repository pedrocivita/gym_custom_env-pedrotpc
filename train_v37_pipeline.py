#
# v3.7 curriculum pipeline. Trains from scratch on 5x5, transfers to
# 10x10 and 20x20 with the new agent vec (10 floats including the
# nearest-unvisited compass added in v3.7).
#
# Why a new pipeline instead of reusing train_curriculum_pipeline.py:
#   - the v3.7 env has a different agent obs shape (10 vs 7), so the
#     pre-existing v3.6 5x5 model cannot be transferred. We have to
#     retrain from scratch.
#   - the v3.6 transfer used ent_coef=0.05 which was diagnosed as the
#     main reason 10x10 stalled at 67% (policy too uniform; could not
#     commit to closing the last few cells). v3.7 uses ent_coef=0.01
#     across all stages so the policy stays decisive throughout.
#
# Stage budgets:
#   5x5  -> 1.0M tsteps (scratch, ent=0.01)
#   10x10 -> 3.0M tsteps (transfer, ent=0.01, lr=1e-4)
#   20x20 -> 1.5M tsteps (transfer, ent=0.01, lr=5e-5)
# Total: 5.5M tsteps, ETA ~70 min on Pedro's CPU.
#

import argparse
import time
from datetime import datetime, timedelta

from train_grid_world_cpp import register_env, train_mode, curriculum_mode, test_mode


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
        "fps_total": 1800,
        "mode": "transfer",
        "lr": 1e-4,
        "ent_coef": 0.01,
    },
    {
        "size": 20, "obstacles": 48, "max_steps": 2000,
        "total_timesteps": 1_500_000,
        "fps_total": 1100,
        "mode": "transfer",
        "lr": 5e-5,
        "ent_coef": 0.01,
    },
]


def fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def print_plan():
    print("=" * 60)
    print("v3.7 Curriculum Pipeline (compass + low-entropy curriculum)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    cumulative = 0.0
    for s in STAGES:
        eta = s["total_timesteps"] / s["fps_total"]
        cumulative += eta
        mode = s["mode"]
        extra = (f" lr={s.get('lr','-')} ent={s.get('ent_coef','-')}"
                 if mode == "transfer" else "")
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-20", action="store_true",
                        help="Stop after 10x10 (skip the 20x20 stretch goal)")
    args = parser.parse_args()

    print_plan()
    if args.dry_run:
        return

    register_env()
    overall_start = time.time()
    results = {}
    last_model_path: str | None = None

    for stage in STAGES:
        s = stage["size"]
        if args.skip_20 and s == 20:
            continue
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
    print(f"v3.7 pipeline finished in {fmt_eta(total_elapsed)}")
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
