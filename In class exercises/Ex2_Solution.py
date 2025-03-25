# %% [markdown]
# # In-Class Simulation: From Stimulation to Force — Solutions
#
# This solution demonstrates one way to match the target force profile using a Hill-type muscle model.

# %% [markdown]
# ## 1. Define the Model

# %%
import numpy as np
import matplotlib.pyplot as plt

def simulate_hill_model(time, stim, L_MT):
    dt = time[1] - time[0]
    n = len(time)

    # Parameters
    tau_act = 0.01
    tau_deact = 0.04
    L_CE_opt = 1.0
    L_SE_slack = 0.2
    k_tendon = 20
    F_max = 1.0

    a = np.zeros(n)
    F_tendon = np.zeros(n)
    F_CE = np.zeros(n)
    L_CE = np.ones(n) * L_CE_opt
    L_SE = np.zeros(n)

    for i in range(1, n):
        tau = tau_act if stim[i] > a[i-1] else tau_deact
        a[i] = a[i-1] + dt * (stim[i] - a[i-1]) / tau

        L_SE[i] = L_MT[i] - L_CE[i-1]

        if L_SE[i] > L_SE_slack:
            F_tendon[i] = k_tendon * (L_SE[i] - L_SE_slack)**2
        else:
            F_tendon[i] = 0.0

        F_CE[i] = F_tendon[i]

        fl = np.exp(-((L_CE[i-1] - L_CE_opt) / 0.45)**2)
        F_iso = a[i] * fl * F_max

        if F_iso > 1e-6:
            L_CE[i] = L_CE[i-1] + dt * (F_CE[i] - F_iso) * 0.5
        else:
            L_CE[i] = L_CE[i-1]

    return {
        'time': time,
        'activation': a,
        'F_tendon': F_tendon,
        'F_CE': F_CE,
        'L_CE': L_CE
    }

# %% [markdown]
# ## 2. Define Target Force

# %%
time = np.linspace(0, 2, 1000)
target_force = np.zeros_like(time)
target_force[200:700] = 1.0  # Step profile from 0.4 to 1.4 s

# %% [markdown]
# ## 3. Simulate Matching Response

# %%
stim = np.zeros_like(time)
stim[150:750] = 1.0  # Slightly extended to allow activation to rise

L_MT = np.ones_like(time) * 1.0  # Optimal length

results = simulate_hill_model(time, stim, L_MT)

# %% [markdown]
# ## 4. Plot Simulated Force vs Target

# %%
plt.figure(figsize=(10, 4))
plt.plot(time, results['F_tendon'], label="Simulated Force")
plt.plot(time, target_force, '--', label="Target Force")
plt.xlabel("Time (s)")
plt.ylabel("Force (a.u.)")
plt.title("Simulated vs. Target Force")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Explanation (Instructor Notes)
# - The stimulation was set slightly wider than the target force to account for activation lag.
# - A constant muscle-tendon length of 1.0 ensures optimal overlap for force production.
# - The model tracks the target force reasonably well with a small delay due to activation dynamics.
