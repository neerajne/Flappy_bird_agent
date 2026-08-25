# 🐦 Flappy Bird AI Agent (Deep Q-Learning)

An autonomous Reinforcement Learning (RL) agent trained to play **Flappy Bird** using **Deep Q-Networks (DQN)** in PyTorch. The agent learns directly from continuous state representations, overcoming core RL challenges like dynamic target shifts and sequential data correlation to consistently score **500+ obstacles passed**.

---

## 📌 Core RL Challenges Solved

Standard Deep Q-Learning in continuous dynamic environments faces major stability issues:

1. **Correlated Sequential Data:** Sequential frames in Flappy Bird are highly correlated, violating the i.i.d. (independent and identically distributed) data assumption required for stable neural network updates.
   * **Solution:** Integrated an **Experience Replay Buffer** to sample mini-batches uniformly from past transitions, breaking temporal correlation.

2. **Moving / Non-Stationary Target Problem:** Updating Q-values against a rapidly changing network leads to target oscillation and training divergence.
   * **Solution:** Used a separate, periodically updated **Target Network** to freeze TD (Temporal Difference) target calculations.

---

## ⚡ Key Features

* **Deep Q-Network Architecture:** Fast and efficient approximation of optimal action-values $Q(s, a)$ using PyTorch.
* **Experience Replay Buffer:** Uniform transition sampling $(s, a, r, s')$ for higher sample efficiency and stable policy convergence.
* **Target Network Stabilization:** Prevents Q-value overestimation and mitigates convergence issues.
* **Model Checkpoint System:** Automatically saves optimal network weights upon reaching new high scores for testing and inference.
* **Dynamic Hyperparameter Tuning:** Configured $\epsilon$-greedy exploration decay, discount factor ($\gamma$), and custom reward structures.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Framework:** PyTorch
* **Environment:** Pygame / Gymnasium
* **Libraries:** NumPy, Matplotlib

---

## 📊 Performance & Results

* **High Score:** **500+** continuous obstacles passed.
* **Stability:** Smooth Q-value stabilization and agent policy convergence achieved during training.

---

## ⚙️ How to Run Locally

```bash
# Clone the repository
git clone [https://github.com/neerajne/Flappy-Bird-DQN.git](https://github.com/neerajne/Flappy-Bird-DQN.git)
cd Flappy-Bird-DQN

# Install dependencies
pip install -r requirements.txt

# Train the agent
python agent.py flappybirdv0 --train

# Evaluate/Test saved model weights
python agent.py flappybirdv0 --train
