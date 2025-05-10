#!/usr/bin/env python3
import numpy as np
import joblib
from Env_Ackermann import AckermannVehicleEnv

# Expert policy parameters
K_V = 1.0                 # velocity gain
K_S = 2.0                 # steering gain
MAX_ACCEL = 1.0           # max acceleration
MAX_STEERING = np.radians(30)  # max steering angle (rad)
AVOID_DISTANCE = 20.0      # threshold for obstacle avoidance (m)
AVOID_STRENGTH = np.radians(20) # additional steering when avoiding (rad)


def expert_policy(obs, env):
    """
    A simple expert controller using proportional control to the goal
    and reactive obstacle avoidance based on LiDAR readings.
    """
    # State encoding: obs[4] = dist_to_goal, obs[5] = bearing_to_goal (rad)
    dist_to_goal = obs[4]
    bearing = obs[5]

    # Proportional control towards goal
    accel = K_V * dist_to_goal
    accel = np.clip(accel, -MAX_ACCEL, MAX_ACCEL)
    steering = K_S * bearing
    steering = np.clip(steering, -MAX_STEERING, MAX_STEERING)

    # Obstacle avoidance: if any LiDAR reading is too close, steer away
    lidar = obs[6:]
    min_dist = np.min(lidar)
    if min_dist < AVOID_DISTANCE:
        # Identify the beam with the closest obstacle
        idx = int(np.argmin(lidar))
        # Compute beam angle in radians (centered around vehicle heading)
        span = env.lidar_span  # in degrees
        beams = np.linspace(-span/2, span/2, env.num_lidar_beams)
        beam_angle = np.radians(beams[idx])
        # Steer away: if obstacle is on left (positive angle), steer right, and vice versa
        steering -= np.sign(beam_angle) * AVOID_STRENGTH
        steering = np.clip(steering, -MAX_STEERING, MAX_STEERING)

    return np.array([accel, steering], dtype=np.float32)


def generate_trajectories(env, num_episodes=100, max_steps=None):
    """
    Collects expert trajectories by running the expert policy in the environment.
    Returns a list of trajectories, each as a list of (state, action) tuples.
    """
    demos = []
    success = 0
    max_steps = max_steps or env.max_episode_steps
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        traj = []
        for t in range(max_steps):
            action = expert_policy(obs, env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            traj.append((obs, action))
            obs = next_obs
            if (info['goal_reached'] == np.True_):
                success+=1
            #if (terminated or truncated) and (info['goal_reached'] == np.False_):
                #demos.append(traj)
            if terminated or truncated:
                break
        demos.append(traj)
        print(f"Episode {ep+1}/{num_episodes} collected with {len(traj)} steps.")
    print(success)
    return demos


if __name__ == '__main__':
    # Instantiate environment
    env = AckermannVehicleEnv()
    # Generate and save expert trajectories
    trajectories = generate_trajectories(env, num_episodes=500)
    joblib.dump(trajectories, 'expert_trajectories.pkl')
    print("Saved expert trajectories to 'expert_trajectories.pkl'")