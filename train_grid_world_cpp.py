#
# Usage:
#   python train_grid_world_cpp.py train <dim> <obstacles> <max_steps> <total_timesteps>
#   python train_grid_world_cpp.py test <dim> <obstacles> [model_path]
#   python train_grid_world_cpp.py run <dim> <obstacles> [model_path]
#   python train_grid_world_cpp.py curriculum <dim> <obstacles> <max_steps> <total_timesteps> <base_model_path>
#

import sys
import os
import glob
import gymnasium as gym
import numpy as np
from datetime import datetime

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env

from utils.feature_extractor import CustomCombinedExtractor


def print_action(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")


def find_latest_model(dim, obstacles):
    pattern = f"data/ppo_cpp_{dim}_{obstacles}_*.zip"
    files = glob.glob(pattern)
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


def make_single_env(dim, obstacles, max_steps, render_mode="rgb_array"):
    return gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        render_mode=render_mode,
    )


def get_model_config(features_dim=128):
    return {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        "shared_lstm": False,
        "enable_critic_lstm": True,
    }


def create_model(env, ent_coef=0.05):
    return RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=get_model_config(),
        device="auto",
    )


def train_mode(dim, obstacles, max_steps, total_timesteps):
    print(f"--- Training CPP Agent: {dim}x{dim}, {obstacles} obstacles, {max_steps} max steps ---")

    env_kwargs = {
        "size": dim,
        "obs_quantity": obstacles,
        "max_steps": max_steps,
        "render_mode": "rgb_array",
    }

    register_env()

    single_env = make_single_env(dim, obstacles, max_steps)
    check_env(single_env)
    single_env.close()

    n_envs = min(8, os.cpu_count() or 4)
    print(f"Using {n_envs} parallel environments")

    env = make_vec_env(
        "gymnasium_env/GridWorldCPP-v0",
        n_envs=n_envs,
        env_kwargs=env_kwargs,
    )

    model = create_model(env)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"log/ppo_cpp_{dim}_{obstacles}_{max_steps}_{timestamp}"
    model_path = f"data/ppo_cpp_{dim}_{obstacles}_{max_steps}_{timestamp}"

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


def curriculum_mode(dim, obstacles, max_steps, total_timesteps, base_model_path):
    print(f"--- Curriculum Learning: {dim}x{dim}, {obstacles} obstacles ---")
    print(f"Loading base model from {base_model_path}")

    env_kwargs = {
        "size": dim,
        "obs_quantity": obstacles,
        "max_steps": max_steps,
        "render_mode": "rgb_array",
    }

    register_env()

    n_envs = min(8, os.cpu_count() or 4)
    print(f"Using {n_envs} parallel environments")

    env = make_vec_env(
        "gymnasium_env/GridWorldCPP-v0",
        n_envs=n_envs,
        env_kwargs=env_kwargs,
    )

    model = RecurrentPPO.load(
        base_model_path,
        env=env,
        device="auto",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"log/ppo_cpp_{dim}_{obstacles}_{max_steps}_{timestamp}_curriculum"
    model_path = f"data/ppo_cpp_{dim}_{obstacles}_{max_steps}_{timestamp}_curriculum"

    os.makedirs("log", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Starting curriculum training: {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    print(f"Logs saved to {log_dir}")

    env.close()
    return model_path


def test_mode(dim, obstacles, model_path):
    max_steps = dim * dim * 4
    print(f"--- Testing CPP Agent: {dim}x{dim}, {obstacles} obstacles, max_steps={max_steps} ---")
    print(f"Loading model from {model_path}")

    register_env()
    model = RecurrentPPO.load(model_path)

    env = make_single_env(dim, obstacles, max_steps, render_mode="rgb_array")

    num_episodes = 100
    full_coverage_count = 0
    total_coverages = []
    total_steps_list = []

    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        lstm_states = None
        episode_start = np.array([True])

        while not done and not truncated:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            obs, reward, done, truncated, info = env.step(action.item())
            episode_start = np.array([False])
            steps += 1

        total_coverages.append(info["coverage"])
        total_steps_list.append(steps)

        if done and not truncated:
            full_coverage_count += 1
            print(f"Episode {i+1}: Full coverage in {steps} steps.")
        else:
            print(f"Episode {i+1}: Coverage {info['coverage']:.1%} in {steps} steps.")

    full_coverage_rate = (full_coverage_count / num_episodes) * 100
    avg_coverage = np.mean(total_coverages) * 100
    std_coverage = np.std(total_coverages) * 100
    avg_steps = np.mean(total_steps_list)
    std_steps = np.std(total_steps_list)

    print(f"\n--- Test Results ({dim}x{dim}) ---")
    print(f"Full Coverage Rate: {full_coverage_rate:.1f}% ({full_coverage_count}/{num_episodes})")
    print(f"Average Coverage: {avg_coverage:.1f}% (std: {std_coverage:.1f}%)")
    print(f"Coverage Range: [{np.min(total_coverages)*100:.1f}%, {np.max(total_coverages)*100:.1f}%]")
    print(f"Average Steps: {avg_steps:.1f} (std: {std_steps:.1f})")
    print(f"Steps Range: [{np.min(total_steps_list)}, {np.max(total_steps_list)}]")

    env.close()
    return {
        "full_coverage_rate": full_coverage_rate,
        "avg_coverage": avg_coverage,
        "std_coverage": std_coverage,
        "avg_steps": avg_steps,
    }


def run_mode(dim, obstacles, model_path):
    max_steps = dim * dim * 4
    print(f"--- Running CPP Agent: {dim}x{dim}, {obstacles} obstacles ---")
    print(f"Loading model from {model_path}")

    register_env()
    model = RecurrentPPO.load(model_path)

    env = make_single_env(dim, obstacles, max_steps, render_mode="human")

    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0
    total_reward = 0
    lstm_states = None
    episode_start = np.array([True])

    while not done and not truncated:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start, deterministic=True
        )
        obs, reward, done, truncated, info = env.step(action.item())
        episode_start = np.array([False])
        total_reward += reward
        steps += 1
        print(
            f"Step: {steps}, Action: {print_action(action.item())}, "
            f"Reward: {reward:.2f}, Coverage: {info['coverage']:.1%}, "
            f"Done: {done}, Truncated: {truncated}"
        )

    print(f"--- Run Finished --- Total reward: {total_reward:.2f}, Coverage: {info['coverage']:.1%}")
    env.close()


def print_usage():
    print("Usage:")
    print("  python train_grid_world_cpp.py train <dim> <obstacles> <max_steps> <total_timesteps>")
    print("  python train_grid_world_cpp.py test <dim> <obstacles> [model_path]")
    print("  python train_grid_world_cpp.py run <dim> <obstacles> [model_path]")
    print("  python train_grid_world_cpp.py curriculum <dim> <obstacles> <max_steps> <total_timesteps> <base_model_path>")


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

    elif mode == "curriculum":
        if len(sys.argv) != 7:
            print_usage()
            sys.exit(1)
        dim = int(sys.argv[2])
        obstacles = int(sys.argv[3])
        max_steps = int(sys.argv[4])
        total_timesteps = int(sys.argv[5])
        base_model_path = sys.argv[6]
        curriculum_mode(dim, obstacles, max_steps, total_timesteps, base_model_path)

    else:
        print_usage()
        sys.exit(1)
