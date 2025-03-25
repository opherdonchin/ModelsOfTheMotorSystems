# %% [markdown]
# # In-Class Simulation: From Stimulation to Force
#
# In this exercise, you'll explore how stimulation and muscle-tendon length affect muscle force,
# using a Hill-type muscle model. Your goal is to match a target force trace by adjusting the input conditions.
#
# You'll use a prebuilt Hill model and investigate how activation dynamics, tendon elasticity,
# and contractile properties interact.

# %% [markdown]
# ## 1. Setup
# Run this cell to import the simulation function from Josh Cashaback's model.

# %%
# Import the prebuilt model
from Hill_type_model_02 import simulate_hill_model
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 2. Define a Target Force Profile
# This is the target force profile you should try to match using stimulation and length inputs.

# %%
# Target force: plateau profile
time = np.linspace(0, 2, 1000)
target_force = np.zeros_like(time)
target_force[200:700] = 1.0  # Simulate a plateau from 0.4s to 1.4s

plt.plot(time, target_force, label="Target Force")
plt.title("Target Force Profile")
plt.xlabel("Time (s)")
plt.ylabel("Force (a.u.)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 3. Try Your Own Inputs
# Adjust the stimulation and muscle-tendon length to try to match the target force profile.

# %%
# Example input: stimulation and MTU length
stim = np.zeros_like(time)
stim[200:700] = 1.0  # Step stimulation

L_MT = np.ones_like(time) * 1.0  # Constant muscle-tendon length

# Simulate the model
results = simulate_hill_model(time, stim, L_MT)

# Plot force output
plt.plot(time, results['F_total'], label="Simulated Force")
plt.plot(time, target_force, '--', label="Target Force")
plt.title("Simulated vs. Target Force")
plt.xlabel("Time (s)")
plt.ylabel("Force (a.u.)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 4. Reflect
# - What changes to the stimulation pattern affected the shape or timing of the force?
# - How did the muscle-tendon length influence the total force?
# - Could you get the force to match the target?
#
# ### Submit:
# - The stimulation and length arrays you used.
# - The final force plot.
# - A short explanation of how you matched the target and what you learned.

