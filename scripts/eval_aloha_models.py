from aloha_wrappers import generate_steerable_aloha_gym_env
import numpy as np
from gymnasium.wrappers import RescaleAction, TimeLimit 
from stable_baselines3 import SAC
import os
import imageio

def main():
    # make env
    ckpt_name = "breezy_lake_5000"
    sac_path = f"./my_models/{ckpt_name}.zip"
    desired_action_dim = 14
    max_timesteps = 400
    seeds = [0]*10
    deterministic = False
    video_dir_path = f"eval_aloha_ckpt{ckpt_name}_determinstic{deterministic}"
    fps=30
    
    os.makedirs(video_dir_path, exist_ok=True)
    
    env = generate_steerable_aloha_gym_env("cuda",desired_action_dim)
    # make action min/max -1,1 based on desired_action_dim
    action_min = np.ones(desired_action_dim)*-1
    action_max = np.ones(desired_action_dim)
    # linearlly normalize obs/action to [-1,1]
    env = RescaleAction(env, min_action=action_min, max_action=action_max)
    wrapped_env = TimeLimit(env, max_episode_steps=max_timesteps)

    # Load the sac model

    model = SAC.load(sac_path,env=wrapped_env)
    
    max_frames = max_timesteps
    all_rollout_frames = []
    all_episode_returns = []

    for rollout in range(len(seeds)):
        obs, info = wrapped_env.reset()
        total_rewards = 0
        # get the initial frame
        frames = wrapped_env.get_wrapper_attr("frames")
        for current_step in range(max_frames):
            # Action select
            # Random action
            # action is shape (batch_size, horizon, action dim)
            if sac_path is not None:
                action = model.predict(obs, deterministic=deterministic)[0]
            else:
                action = wrapped_env.action_space.sample()
            # Step in env in wrapper model.
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_rewards += reward
            frames += wrapped_env.get_wrapper_attr("frames")

            if terminated or truncated:
                break
        print(f"terminated:{terminated}, returns: {total_rewards}")
        
        # get frames
        # Save individual video
        video_path = f"{video_dir_path}/episode_{rollout}_seed_{seeds[rollout]}.mp4"
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Saved {video_path}, terminated:{terminated}, returns: {total_rewards}")

        # save the returns
        all_episode_returns.append(total_rewards)
        np.save(f"{video_dir_path}/returns.npy", all_episode_returns)


if __name__ == "__main__":
    main()