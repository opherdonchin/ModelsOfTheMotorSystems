# %% [markdown]
# # Homework 1: Modeling Muscle Behavior with Hill-Type Models
#
# **Instructions:** Complete all sections below. You will build and explore a Hill-type muscle model
# to simulate twitch, tonic, and shortening contractions. Answer all written questions in the appropriate
# markdown sections. Submit your completed `.ipynb` notebook on Moodle.

# %% [markdown]
# ## Part A: Build Your Hill-Type Muscle Model
#
# Implement a function called `simulate_hill_model()` that models muscle force output over time.
# Your model **must include**:
# - Activation dynamics (with different activation and deactivation time constants)
# - Active force–length relationship (e.g. Gaussian)
# - Force–velocity relationship (may be simplified)
# - Passive muscle force
# - Nonlinear tendon elasticity
#
# This function will be used throughout the rest of the assignment.

# %%
# TODO: Define simulate_hill_model(time, stim, L_MT)
# Inputs:
#   - time: time array
#   - stim: stimulation signal (0–1)
#   - L_MT: muscle-tendon unit length array
# Returns:
#   - Dictionary with 'activation', 'F_total', 'F_tendon', 'L_CE', etc.

# %% [markdown]
# ## Part B1: Simulate a Twitch Contraction
#
# Use a brief stimulation pulse (~5 ms) to elicit a twitch response.
# Plot the following over time:
# - Activation
# - Muscle force
# - Fiber length (L_CE)
# - Tendon length (L_SE)
#
# Use this to verify that your model responds correctly to a short pulse.

# %%
# TODO: Define stimulation signal (short pulse)
# TODO: Choose reasonable MTU length (e.g. L_MT = 1.0)
# TODO: Simulate and plot the outputs

# %% [markdown]
# ### Answer B1:
# - What was the peak force?
# - What was the time to half-relaxation?
# - How would changing deactivation time constant affect this?

# %% [markdown]
# ## Part B2: Tonic Contractions at Different Lengths
#
# Apply constant stimulation for 1 second. Simulate the muscle at **three different lengths**
# (e.g. L_MT = 0.95, 1.0, and 1.05). Plot steady-state force vs. muscle-tendon length.
# This explores the active force–length curve.

# %%
# TODO: Loop over multiple MTU lengths
# TODO: Simulate and extract steady-state force for each
# TODO: Plot force vs. length

# %% [markdown]
# ### Answer B2:
# - Describe the trend you observed.
# - How does this relate to the active force–length relationship discussed in lecture?

# %% [markdown]
# ## Part B3: Simulate a Shortening Contraction
#
# Create a simulation where the muscle shortens over time during stimulation.
# - Use a ramped or step-decreasing L_MT.
# - Keep stimulation constant.
# - Plot force vs. velocity.
#
# Compare the resulting curve to the theoretical force–velocity relationship.

# %%
# TODO: Define time-varying L_MT (e.g. linear ramp)
# TODO: Simulate and calculate instantaneous velocity
# TODO: Plot force vs. velocity

# %% [markdown]
# ### Answer B3:
# - How does the simulated curve compare to the idealized force–velocity plot?
# - What factors limit your model’s ability to match the theoretical curve?

# %% [markdown]
# ## ⭐ Part D (Bonus +3%): Simulate an Older Muscle
#
# Modify your model to simulate behavior of an older adult muscle based on Thelen (2003).
# Specifically:
# - Increase deactivation time constant
# - Reduce maximum isometric force (Fmax)
# - Increase passive stiffness (if implemented)
#
# Simulate both a twitch and tonic contraction and compare to your original results.

# %%
# TODO: Modify model parameters to reflect aging
# TODO: Repeat twitch and tonic simulations
# TODO: Plot and compare to young muscle

# %% [markdown]
# ### Answer D:
# - What differences did you observe in timing, magnitude, or shape of the responses?
# - Which parameter had the most noticeable effect?
