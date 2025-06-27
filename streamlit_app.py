import streamlit as st
import gym
import numpy as np
import imageio

# Create CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Run one episode with random actions
def run_cartpole():
    frames = []
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        frames.append(env.render())
        action = env.action_space.sample()  # Random actions
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    env.close()

    gif_path = "cartpole_run.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    return gif_path, total_reward

# Streamlit UI
st.title("🎮 CartPole RL Demo App")
st.write("Click the button to watch the agent balance the pole.")

if st.button("▶ Run Episode"):
    with st.spinner("Running..."):
        gif, score = run_cartpole()
        st.image(gif, caption=f"Score: {score}", use_column_width=True)
