#
# Coverage Path Planning trainer (v3) — MaskablePPO + visit-map observation.
#
# Usage:
#   python train_grid_world_cpp.py train <dim> <obstacles> <max_steps> <total_timesteps>
#   python train_grid_world_cpp.py test  <dim> <obstacles> [model_path]
#   python train_grid_world_cpp.py run   <dim> <obstacles> [model_path]
#

import sys
import os
import glob
import gymnasium as gym
import numpy as np
from datetime import datetime

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env

from utils.feature_extractor import CustomCombinedExtractor


ACTION_NAMES = {0: "right", 1: "up", 2: "left", 3: "down"}


def print_action(action: int) -> str:
    return ACTION_NAMES.get(action, "unknown")


def find_latest_model(dim, obstacles):
    pattern = f"data/maskppo_cpp_{dim}_{obstacles}_*.zip"
    files = glob.glob(pattern)
    if not files:
        # Backwards compat: also look for older naming used during refactor.
        files = glob.glob(f"data/ppo_cpp_{dim}_{obstacles}_*.zip")
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def register_env():
    try:
        gym.register(
            id="gymnasium_env/GridWorldCPP-v0",
            entry_point=GridWorldCPPEnv,
        )
    except Exception:
        pass


def _mask_fn(env: gym.Env) -> np.ndarray:
    # ActionMasker calls this to query legal actions; unwrap to reach our env.
    return env.unwrapped.action_masks()


def make_single_env(dim, obstacles, max_steps, render_mode="rgb_array"):
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        render_mode=render_mode,
    )
    return ActionMasker(env, _mask_fn)


def _make_env_factory(dim, obstacles, max_steps):
    def _factory():
        return make_single_env(dim, obstacles, max_steps)
    return _factory


def get_policy_kwargs(features_dim=128):
    return {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
    }


def create_model(env, ent_coef=0.01):
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=get_policy_kwargs(),
        device="cpu",
    )


def train_mode(dim, obstacles, max_steps, total_timesteps):
    print(f"--- Training CPP Agent: {dim}x{dim}, {obstacles} obstacles, {max_steps} max steps ---")

    register_env()

    # Pre-flight check on a raw env (before ActionMasker wrap).
    raw_env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        render_mode="rgb_array",
    )
    check_env(raw_env.unwrapped)
    raw_env.close()

    n_envs = min(8, os.cpu_count() or 4)
    print(f"Using {n_envs} parallel environments")

    env_fns = [_make_env_factory(dim, obstacles, max_steps) for _ in range(n_envs)]
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv(env_fns)

    model = create_model(env)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"log/maskppo_cpp_{dim}_{obstacles}_{max_steps}_{timestamp}"
    model_path = f"data/maskppo_cpp_{dim}_{obstacles}_{max_steps}_{timestamp}"

    os.makedirs("log", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Starting training: {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    print(f"Logs saved to {log_dir}")

    env.close()
    return model_path


def _run_test_episodes(env, model, num_episodes: int, deterministic: bool, label: str):
    full_coverage_count = 0
    total_coverages = []
    total_steps_list = []

    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0

        while not done and not truncated:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(int(action))
            steps += 1

        total_coverages.append(info["coverage"])
        total_steps_list.append(steps)

        if done and not truncated:
            full_coverage_count += 1

    full_coverage_rate = (full_coverage_count / num_episodes) * 100
    avg_coverage = float(np.mean(total_coverages) * 100)
    std_coverage = float(np.std(total_coverages) * 100)
    avg_steps = float(np.mean(total_steps_list))
    std_steps = float(np.std(total_steps_list))

    print(f"\n--- {label} ({deterministic=}) ---")
    print(f"Full Coverage Rate: {full_coverage_rate:.1f}% ({full_coverage_count}/{num_episodes})")
    print(f"Average Coverage: {avg_coverage:.1f}% (std: {std_coverage:.1f}%)")
    print(f"Coverage Range: [{np.min(total_coverages)*100:.1f}%, {np.max(total_coverages)*100:.1f}%]")
    print(f"Average Steps: {avg_steps:.1f} (std: {std_steps:.1f})")
    print(f"Steps Range: [{np.min(total_steps_list)}, {np.max(total_steps_list)}]")

    return {
        "full_coverage_rate": full_coverage_rate,
        "avg_coverage": avg_coverage,
        "std_coverage": std_coverage,
        "avg_steps": avg_steps,
        "std_steps": std_steps,
    }


def test_mode(dim, obstacles, model_path, num_episodes: int = 100):
    max_steps = dim * dim * 4
    print(f"--- Testing CPP Agent: {dim}x{dim}, {obstacles} obstacles, max_steps={max_steps} ---")
    print(f"Loading model from {model_path}")

    register_env()
    model = MaskablePPO.load(model_path)

    env = make_single_env(dim, obstacles, max_steps, render_mode="rgb_array")

    det = _run_test_episodes(env, model, num_episodes, True,  f"Deterministic ({dim}x{dim})")
    sto = _run_test_episodes(env, model, num_episodes, False, f"Stochastic    ({dim}x{dim})")

    env.close()
    print(f"\n=== Summary {dim}x{dim} ===")
    print(f"Deterministic: full={det['full_coverage_rate']:.1f}%  avg={det['avg_coverage']:.1f}%  steps={det['avg_steps']:.0f}")
    print(f"Stochastic:    full={sto['full_coverage_rate']:.1f}%  avg={sto['avg_coverage']:.1f}%  steps={sto['avg_steps']:.0f}")

    return {"deterministic": det, "stochastic": sto}


def run_mode(dim, obstacles, model_path):
    max_steps = dim * dim * 4
    print(f"--- Running CPP Agent: {dim}x{dim}, {obstacles} obstacles ---")
    print(f"Loading model from {model_path}")

    register_env()
    model = MaskablePPO.load(model_path)

    env = make_single_env(dim, obstacles, max_steps, render_mode="human")

    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0
    total_reward = 0

    while not done and not truncated:
        masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        print(
            f"Step: {steps}, Action: {print_action(int(action))}, "
            f"Reward: {reward:.2f}, Coverage: {info['coverage']:.1%}, "
            f"Done: {done}, Truncated: {truncated}"
        )

    print(f"--- Run Finished --- Total reward: {total_reward:.2f}, Coverage: {info['coverage']:.1%}")
    env.close()


def print_usage():
    print("Usage:")
    print("  python train_grid_world_cpp.py train <dim> <obstacles> <max_steps> <total_timesteps>")
    print("  python train_grid_world_cpp.py test  <dim> <obstacles> [model_path]")
    print("  python train_grid_world_cpp.py run   <dim> <obstacles> [model_path]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "train":
        if len(sys.argv) != 6:
            print_usage()
            sys.exit(1)
        dim = int(sys.argv[2])
        obstacles = int(sys.argv[3])
        max_steps = int(sys.argv[4])
        total_timesteps = int(sys.argv[5])
        train_mode(dim, obstacles, max_steps, total_timesteps)

    elif mode == "test":
        if len(sys.argv) < 4:
            print_usage()
            sys.exit(1)
        dim = int(sys.argv[2])
        obstacles = int(sys.argv[3])
        model_path = sys.argv[4] if len(sys.argv) > 4 else find_latest_model(dim, obstacles)
        if model_path is None:
            print(f"No model found for {dim}x{dim} with {obstacles} obstacles. Train first.")
            sys.exit(1)
        test_mode(dim, obstacles, model_path)

    elif mode == "run":
        if len(sys.argv) < 4:
            print_usage()
            sys.exit(1)
        dim = int(sys.argv[2])
        obstacles = int(sys.argv[3])
        model_path = sys.argv[4] if len(sys.argv) > 4 else find_latest_model(dim, obstacles)
        if model_path is None:
            print(f"No model found for {dim}x{dim} with {obstacles} obstacles. Train first.")
            sys.exit(1)
        run_mode(dim, obstacles, model_path)

    else:
        print_usage()
        sys.exit(1)
