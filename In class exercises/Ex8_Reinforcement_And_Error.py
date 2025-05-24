# %% [markdown]
# # Reward and Error Learning Simulation (Izawa & Shadmehr, 2011)

# %% [markdown]
# ## 1. Imports and Constants

# %%
import numpy as np
import matplotlib.pyplot as plt

# Simulation settings
n_trials = 200
gamma = 0.9           # discount factor
alpha_r = 0.1         # learning rate for reward prediction error
alpha_v = 0.1         # learning rate for value function
a = 0.98              # forgetting factor for perturbation
b = np.array([[0], [1]])  # input matrix
C = np.array([[0, 1]])    # observation matrix
A = np.array([[a, 0], [1, 0]])

# Initial values
x = np.zeros((2, 1))    # true state: [p; h]
x_hat = np.zeros((2, 1)) # state estimate
w_r = 0.0                # reward-based controller
w_v = 0.0                # value estimate
nu = 0.0                 # motor noise

# %% [markdown]
# ## 2. Task Environment Setup

# %%
def simulate(condition="ERR", seed=42):
    np.random.seed(seed)
    # Visual noise level varies by condition
    sigma_y = {
        "ERR": 5.0,
        "EPE": 20.0,
        "RWD": 1e6  # essentially no observation
    }[condition]
    
    P = np.diag([1.0, 1.0])
    Q = np.diag([0.05, 0.05])
    R = sigma_y ** 2

    h_history = []
    h_hat_history = []

    p_hat = 0.0
    w_r = 0.0
    w_v = 0.0

    for k in range(n_trials):
        # true perturbation dynamics
        p = a * x[0, 0] + np.random.normal(0, Q[0, 0]**0.5)
        u = -p_hat + w_r + np.random.normal(0, 0.5)  # control policy
        h = u + np.random.normal(0, Q[1, 1]**0.5)

        x = np.array([[p], [h]])
        y = C @ x + np.random.normal(0, R**0.5, size=(1, 1))

        # Kalman prediction
        x_hat = A @ x_hat + b * u
        P = A @ P @ A.T + Q

        # Kalman update
        K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
        x_hat = x_hat + K @ (y - C @ x_hat)
        P = (np.eye(2) - K @ C) @ P

        # reward computation
        reward = 1.0 if abs(h) < 3.0 else 0.0
        delta_r = reward + gamma * w_v - w_v
        w_v += alpha_v * delta_r
        w_r += alpha_r * delta_r * nu

        h_history.append(h)
        h_hat_history.append(x_hat[1, 0])

    return np.array(h_history), np.array(h_hat_history)

# %% [markdown]
# ## 3. Run Simulation for Each Condition

# %%
conditions = ["ERR", "EPE", "RWD"]
results = {}

for cond in conditions:
    h, h_hat = simulate(cond)
    results[cond] = (h, h_hat)

# %% [markdown]
# ## 4. Plotting Estimated vs Actual Hand Position

# %%
plt.figure(figsize=(12, 6))
for cond, (h, h_hat) in results.items():
    plt.plot(h_hat, label=f"{cond} (estimate)")
plt.plot(results["ERR"][0], '--', color='k', label="True hand (ERR)")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Estimated Hand Position Across Conditions")
plt.xlabel("Trial")
plt.ylabel("Hand Position (deg)")
plt.legend()
plt.grid(True)
plt.show()
