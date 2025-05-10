#!/usr/bin/env python3


import argparse
import os
import numpy as np
import joblib
import torch
import pyro
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from Env_Ackermann import AckermannVehicleEnv  
from BNN_v2 import BNN                         

class BNNRewardEnv(gym.Wrapper):
   
    def __init__(self, env, predictive, scaler, beta=0.1, mode="explore", samples_per_forward=30):
        super().__init__(env)
        assert mode in {"explore", "risk_averse", "none"}
        self.predictive = predictive
        self.scaler = scaler
        self.beta = float(beta)
        self.mode = mode
        self.spp = samples_per_forward

    def set_uncertainty_params(self, *, beta=None, mode=None):
        if beta is not None:
            self.beta = float(beta)
        if mode is not None:
            assert mode in {"explore", "risk_averse", "none"}
            self.mode = mode

    def _features(self, obs: np.ndarray, action: np.ndarray):
        s = np.asarray(obs, dtype=np.float32)
        dist_goal = s[4:5]
        lidar = s[6:]
        feats = np.concatenate([
            dist_goal,
            np.min(lidar, keepdims=True),
            np.mean(lidar, keepdims=True),
            np.std(lidar, keepdims=True),
            np.asarray([action[1]], dtype=np.float32),  # steering angle
        ])
        x_scaled = self.scaler.transform(feats.reshape(1, -1))
        return torch.from_numpy(x_scaled).float()

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        with torch.no_grad():
            x = self._features(obs, action)
            preds = self.predictive(x)["obs"]  # (samples, 1)
            mu = preds.mean(0).item()
            sigma = preds.std(0).item()
        if self.mode == "explore":
            reward = mu + self.beta * sigma
        elif self.mode == "risk_averse":
            reward = mu - self.beta * sigma
        else:  # none
            reward = mu
        info.update({"reward_mu": mu, "reward_sigma": sigma})
        return obs, reward, terminated, truncated, info


def make_vec_env(predictive, scaler, beta, mode):
    base_env = AckermannVehicleEnv()
    rew_env = BNNRewardEnv(base_env, predictive, scaler, beta=beta, mode=mode)
    return DummyVecEnv([lambda: rew_env]), rew_env  # return underlying wrapper too


def load_bnn(device):
    guide = torch.load("bnn_reward_guide.pt", map_location=device)
    scaler = joblib.load("reward_scaler.pkl")
    input_dim = scaler.mean_.shape[0]
    bnn = BNN(input_dim=input_dim).to(device)
    predictive = pyro.infer.Predictive(bnn, guide=guide, return_sites=["obs"], num_samples=30)
    return predictive, scaler


def single_stage_training(total_steps, beta, mode, device, model_path=None, env_wrapper=None, vec_env=None):
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path, env=vec_env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            tensorboard_log="./ppo_bnn_uncertainty_tb",
            verbose=1,
            device=device,
        )
    # ensure env has correct beta/mode
    env_wrapper.set_uncertainty_params(beta=beta, mode=mode)
    model.learn(total_timesteps=total_steps)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO with BNN + uncertainty reward (auto two‑stage capable)")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--mode", choices=["explore", "risk_averse", "none", "auto_two_stage"], default="auto_two_stage")
    parser.add_argument("--beta", type=float, default=0.1, help="Used when mode != auto_two_stage")
    parser.add_argument("--beta_explore", type=float, default=0.25)
    parser.add_argument("--beta_risk", type=float, default=0.07)
    parser.add_argument("--explore_frac", type=float, default=0.4, help="Fraction of steps in explore stage when mode=auto_two_stage")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    predictive, scaler = load_bnn(device)

    vec_env, rew_wrapper = make_vec_env(predictive, scaler, beta=args.beta, mode="none")

    if args.mode == "auto_two_stage":
        steps_explore = int(args.timesteps * args.explore_frac)
        steps_risk = args.timesteps - steps_explore

        print(f"[Stage 1] explore for {steps_explore:,} steps, beta={args.beta_explore}")
        model = single_stage_training(
            total_steps=steps_explore,
            beta=args.beta_explore,
            mode="explore",
            device=device,
            env_wrapper=rew_wrapper,
            vec_env=vec_env,
        )

        print(f"[Stage 2] risk‑averse for {steps_risk:,} steps, beta={args.beta_risk}")
        model = single_stage_training(
            total_steps=steps_risk,
            beta=args.beta_risk,
            mode="risk_averse",
            device=device,
            env_wrapper=rew_wrapper,
            vec_env=vec_env,
            model_path=None,  # continue from RAM model, no need to reload from disk
        )

        model.save("ppo_bnn_uncertainty_auto_two_stage.zip")
        print("Saved final model to ppo_bnn_uncertainty_auto_two_stage.zip")
    else:
        # single‑mode run
        model = single_stage_training(
            total_steps=args.timesteps,
            beta=args.beta,
            mode=args.mode,
            device=device,
            env_wrapper=rew_wrapper,
            vec_env=vec_env,
        )
        fname = f"ppo_bnn_uncertainty_{args.mode}_beta{args.beta}.zip"
        model.save(fname)
        print(f"Saved model to {fname}")


if __name__ == "__main__":
    main()
