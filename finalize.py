#
# Finalize the APS deliverable.
#
# Auto-detects the most recent trained model for each grid size, runs the
# dual-mode test (100 episodes deterministic + 100 stochastic), substitutes
# the {{...}} placeholders in report_cpp.md, refreshes the summary block in
# README.md, and (optionally) commits and pushes the result.
#
# Usage:
#   python finalize.py            # full pipeline: test all sizes, fill report,
#                                 # refresh README, commit + push.
#   python finalize.py --no-push  # everything except git push.
#   python finalize.py --no-git   # only test + write files, no commit/push.
#   python finalize.py --dry-run  # report what would be done, write nothing.
#
# Idempotent: re-running with new training will overwrite the same blocks.
#

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv  # noqa: F401 (registers env on import path)
from train_grid_world_cpp import register_env, test_mode

REPO_ROOT = Path(__file__).resolve().parent
REPORT_PATH = REPO_ROOT / "report_cpp.md"
README_PATH = REPO_ROOT / "README.md"
RESULTS_JSON = REPO_ROOT / "data" / "finalize_results.json"

# Grid stages we report on. Must match what train_local_pipeline.py used.
SIZES = [
    {"size": 5,  "obstacles": 3,  "tag": "5"},
    {"size": 10, "obstacles": 12, "tag": "10"},
    {"size": 20, "obstacles": 48, "tag": "20"},
]


def find_latest_model(size: int, obstacles: int):
    """Return the newest .zip model file matching the given size+obstacles.

    Looks for the v3.x naming convention; falls back to legacy patterns so a
    user with older artifacts can still run finalize.py.
    """
    candidates = []
    patterns = [
        f"data/maskppo_cpp_{size}_{obstacles}_*.zip",
        f"data/ppo_cpp_{size}_{obstacles}_*.zip",
    ]
    for pat in patterns:
        candidates.extend(glob.glob(str(REPO_ROOT / pat)))
    # Exclude per-step checkpoint zips (they live under *_checkpoints/).
    candidates = [c for c in candidates if "_checkpoints" not in c]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def run_tests(skip_missing: bool = False) -> dict:
    """Run dual-mode test for each size; return a dict keyed by tag."""
    register_env()
    out = {}
    for cfg in SIZES:
        path = find_latest_model(cfg["size"], cfg["obstacles"])
        if path is None:
            msg = f"[WARN] no trained model found for {cfg['size']}x{cfg['size']} ({cfg['obstacles']} obstacles)"
            if skip_missing:
                print(msg + " — skipping")
                out[cfg["tag"]] = None
                continue
            raise FileNotFoundError(msg)
        print(f"\n=== Evaluating {cfg['size']}x{cfg['size']} from {path} ===")
        metrics = test_mode(cfg["size"], cfg["obstacles"], path)
        out[cfg["tag"]] = {"model_path": path, **metrics}
    return out


def _fmt_pct(x):
    if x is None:
        return "—"
    return f"{x:.1f}%"


def _fmt_steps(x):
    if x is None:
        return "—"
    return f"{x:.0f}"


def fill_report(results: dict, dry_run: bool = False) -> str:
    """Substitute {{TAG_MODE_FIELD}} placeholders in report_cpp.md."""
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"{REPORT_PATH} not found")
    content = REPORT_PATH.read_text(encoding="utf-8")

    repl = {}
    for tag in ("5", "10", "20"):
        m = results.get(tag)
        det = m["deterministic"] if m else None
        sto = m["stochastic"] if m else None
        # Full coverage rate
        repl[f"{{{{{tag}_DET_FULL}}}}"]   = _fmt_pct(det["full_coverage_rate"]) if det else "—"
        repl[f"{{{{{tag}_STOCH_FULL}}}}"] = _fmt_pct(sto["full_coverage_rate"]) if sto else "—"
        # Avg coverage
        repl[f"{{{{{tag}_DET_AVG}}}}"]   = _fmt_pct(det["avg_coverage"]) if det else "—"
        repl[f"{{{{{tag}_STOCH_AVG}}}}"] = _fmt_pct(sto["avg_coverage"]) if sto else "—"
        # Avg steps
        repl[f"{{{{{tag}_DET_STEPS}}}}"]   = _fmt_steps(det["avg_steps"]) if det else "—"
        repl[f"{{{{{tag}_STOCH_STEPS}}}}"] = _fmt_steps(sto["avg_steps"]) if sto else "—"

    new_content = content
    for token, value in repl.items():
        new_content = new_content.replace(token, value)

    if not dry_run:
        REPORT_PATH.write_text(new_content, encoding="utf-8")
    return new_content


def refresh_readme_summary(results: dict, dry_run: bool = False):
    """Replace the placeholder rows in the README summary table."""
    if not README_PATH.exists():
        return
    content = README_PATH.read_text(encoding="utf-8")
    # Build a fresh markdown table block.
    rows = []
    for tag, label in (("5", "5×5"), ("10", "10×10"), ("20", "20×20")):
        m = results.get(tag)
        det = m["deterministic"] if m else None
        sto = m["stochastic"] if m else None
        rows.append(
            f"| {label} | {_fmt_pct(det['full_coverage_rate']) if det else '—'} | "
            f"{_fmt_pct(sto['full_coverage_rate']) if sto else '—'} |"
        )

    new_block = (
        "| Grid | Full coverage (deterministic) | Full coverage (stochastic) |\n"
        "|------|-------------------------------|----------------------------|\n"
        + "\n".join(rows)
    )

    pattern = re.compile(
        r"\| Grid \| Full coverage \(deterministic\) \| Full coverage \(stochastic\) \|.*?(?=\n\n|\Z)",
        flags=re.DOTALL,
    )
    if not pattern.search(content):
        print("[WARN] README summary table block not found — skipping README refresh")
        return
    new_content = pattern.sub(new_block, content)
    if not dry_run:
        README_PATH.write_text(new_content, encoding="utf-8")


def save_results_json(results: dict, dry_run: bool = False):
    if dry_run:
        return
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    RESULTS_JSON.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def git_commit_and_push(push: bool = True, dry_run: bool = False):
    msg_lines = [
        "Finalize: fill report and README summary with run results",
        "",
        "Auto-generated by finalize.py — substitutes {{...}} placeholders",
        "in report_cpp.md and refreshes the summary table in README.md.",
    ]
    msg = "\n".join(msg_lines)

    cmds = [
        ["git", "add", "report_cpp.md", "README.md", "data/finalize_results.json"],
        ["git", "commit", "-m", msg],
    ]
    if push:
        cmds.append(["git", "push", "origin", "main"])

    for cmd in cmds:
        printable = " ".join(cmd if cmd[0] != "git" or "commit" not in cmd else cmd[:3] + ["<message>"])
        print(f"$ {printable}")
        if dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            # `git commit` returns non-zero when there is nothing to commit.
            if "nothing to commit" in (result.stdout + result.stderr).lower():
                print("(nothing new to commit)")
                return
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"git command failed: {' '.join(cmd)}")
        if result.stdout:
            print(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-push", action="store_true",
                        help="Commit but don't push to origin.")
    parser.add_argument("--no-git", action="store_true",
                        help="Skip commit and push (only update files).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen, don't write or commit.")
    parser.add_argument("--skip-missing", action="store_true",
                        help="If a model for some size is missing, fill that "
                             "block with em-dashes instead of failing.")
    args = parser.parse_args()

    print("=" * 60)
    print(f"finalize.py — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = run_tests(skip_missing=args.skip_missing)
    print("\n=== Summary ===")
    for tag, m in results.items():
        if m is None:
            print(f"  {tag}x{tag}: NO MODEL")
            continue
        print(
            f"  {tag}x{tag}: det={m['deterministic']['full_coverage_rate']:.1f}%/"
            f"{m['deterministic']['avg_coverage']:.1f}%  "
            f"stoch={m['stochastic']['full_coverage_rate']:.1f}%/"
            f"{m['stochastic']['avg_coverage']:.1f}%"
        )

    fill_report(results, dry_run=args.dry_run)
    refresh_readme_summary(results, dry_run=args.dry_run)
    save_results_json(results, dry_run=args.dry_run)

    if args.dry_run or args.no_git:
        print("\n[done] files updated (no git ops).")
        return

    git_commit_and_push(push=not args.no_push, dry_run=args.dry_run)
    print("\n[done] finalize complete.")


if __name__ == "__main__":
    main()
