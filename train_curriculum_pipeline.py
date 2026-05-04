#
# Automated curriculum learning pipeline: 5x5 -> 10x10 -> 20x20
#
# Usage:
#   python train_curriculum_pipeline.py              (full pipeline)
#   python train_curriculum_pipeline.py --skip-5x5 <model_path>  (start from 10x10)
#

import sys
from train_grid_world_cpp import train_mode, curriculum_mode, test_mode

STAGES = [
    {"size": 5,  "obstacles": 3,  "max_steps": 150,  "timesteps": 3_000_000},
    {"size": 10, "obstacles": 12, "max_steps": 500,  "timesteps": 5_000_000},
    {"size": 20, "obstacles": 48, "max_steps": 2000, "timesteps": 10_000_000},
]


def run_pipeline(start_model=None):
    model_path = start_model
    start_idx = 1 if start_model else 0

    for i, stage in enumerate(STAGES):
        s = stage["size"]
        o = stage["obstacles"]
        ms = stage["max_steps"]
        ts = stage["timesteps"]

        if i < start_idx:
            continue

        print(f"\n{'='*60}")
        print(f"STAGE {i+1}: {s}x{s} grid, {o} obstacles")
        print(f"{'='*60}\n")

        if model_path is None:
            model_path = train_mode(s, o, ms, ts)
        else:
            model_path = curriculum_mode(s, o, ms, ts, model_path)

        print(f"\n--- Evaluating Stage {i+1} ---")
        results = test_mode(s, o, model_path)
        print(f"Stage {i+1} complete: {results['full_coverage_rate']:.1f}% full coverage\n")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if "--skip-5x5" in sys.argv:
        idx = sys.argv.index("--skip-5x5")
        if idx + 1 < len(sys.argv):
            run_pipeline(start_model=sys.argv[idx + 1])
        else:
            print("Usage: python train_curriculum_pipeline.py --skip-5x5 <model_path>")
            sys.exit(1)
    else:
        run_pipeline()
