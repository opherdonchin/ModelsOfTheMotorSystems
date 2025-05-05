# %% [markdown]
# Homework: Assignment 4 Problem 7 & Assignment 2 Q7b

**Instructions:** Work through all parts below. Provide full derivations, code, and figures. All blank-coded sections (`# TODO`) must be completed by you; scaffolding is minimal. You may add helper functions but core implementations must be your own. Include brief explanatory comments and intermediate checks where helpful.

# %% [markdown]
## Part A: Two-Link Arm Dynamics (Assignment 4, Problem 7)

This problem explores planar arm dynamics using the Lagrangian approach.

1. **Derivation (by hand):**
   - Define joint angles $\theta=[\theta_1,\theta_2]^T$. The equations of motion take the form
     $$M(\theta)\ddot{\theta} + C(\theta,\dot{\theta}) + G(\theta)=0,$$
     where:
     - **Mass matrix** $M(\theta)$ includes link inertias and coupling terms.
     - **Coriolis/centrifugal** vector $C(\theta,\dot{\theta})$ captures velocity-dependent forces.
     - **Gravity** vector $G(\theta)$ contains $m_i g r_i \sin(\theta_i)$ contributions.
   - Show all steps, symbol definitions, and final expressions for $M$, $C$, and $G$.
   - Solve algebraically for
     $$\ddot{\theta} = -M^{-1}(\theta) \bigl[C(\theta,\dot{\theta}) + G(\theta)\bigr].$$
   - **Hint:** Verify that your $M$ is symmetric and positive definite.

2. **Implementation:**
   - Set parameters: 
     ```python
     params = dict(
         m1=2.1, m2=1.65,
         I1=0.025, I2=0.075,
         l1=0.3384, l2=0.4554,
         r1=0.1692, r2=0.2277,
         g=9.81
     )
     ```
   - Use Euler integration (time step $\Delta t = 1e\!{-5}$ s) for $t\in[0,2]$ s.
   - Initial states: $\theta(0)=[180°,1°]$ (convert to radians), $\dot{\theta}(0)=[0,0]$.
   - **Strategy:** Implement a function `two_link_eom(theta, dtheta, params)` that returns $\ddot{\theta}$.  Include inline comments explaining terms.

3. **Simulation and Plots:**
   - Run the integration loop.
   - Plot:
     - $\theta_1(t)$, $\theta_2(t)$ in degrees.
     - $\dot{\theta}_1(t)$, $\dot{\theta}_2(t)$ in rad/s.
   - Label axes and add legends.

4. **Energy Analysis:**
   - Define kinetic energy
     $$T = \tfrac12 \dot{\theta}^T M(\theta) \dot{\theta}.$$  
   - Define potential energy (zero at shoulder height)
     $$U = m_1 g r_1 (1-\cos\theta_1) + m_2 g [l_1(1-\cos\theta_1) + r_2(1-\cos(\theta_1+\theta_2))].$$
   - Implement `compute_energies(theta, dtheta, params)` returning arrays `T`, `U`.
   - Plot `T`, `U`, and `E=T+U` versus time.
   - **Discuss:** Does total energy drift? Relate to integration error.

> **Note:** Break derivation and coding into clear sections. Use assertions or small tests (e.g., check symmetry of $M$) to validate intermediate results.

# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters & initial state
dt = 1e-5
time = np.arange(0, 2.0+dt, dt)
theta = np.zeros((len(time), 2))
dtheta = np.zeros_like(theta)
theta[0] = np.deg2rad([180, 1])
dtheta[0] = [0.0, 0.0]
params = dict(
    m1=2.1, m2=1.65,
    I1=0.025, I2=0.075,
    l1=0.3384, l2=0.4554,
    r1=0.1692, r2=0.2277,
    g=9.81
)

# TODO: Implement two_link_eom
# Should compute M, C, G and return -np.linalg.solve(M, C+G)
def two_link_eom(theta, dtheta, p):
    """
    Compute angular accelerations for two-link arm.
    theta: [theta1, theta2]
    dtheta: [dtheta1, dtheta2]
    Returns: [ddtheta1, ddtheta2]
    """
    raise NotImplementedError

# Simulation loop
for i in range(1, len(time)):
    ddth = two_link_eom(theta[i-1], dtheta[i-1], params)
    theta[i]  = theta[i-1]  + dtheta[i-1] * dt
    dtheta[i] = dtheta[i-1] + ddth          * dt

# Plot angles & velocities
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, np.rad2deg(theta[:, 0]), label='$\theta_1$')
plt.plot(time, np.rad2deg(theta[:, 1]), label='$\theta_2$')
plt.ylabel('Angle (deg)'); plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, dtheta[:, 0], label='$\dot\theta_1$')
plt.plot(time, dtheta[:, 1], label='$\dot\theta_2$')
plt.ylabel('Angular vel (rad/s)'); plt.xlabel('Time (s)'); plt.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
## Energy Computation and Analysis

Implement kinetic and potential energies and analyze drift.

# %%
# TODO: Implement compute_energies
def compute_energies(theta, dtheta, p):
    """
    Returns arrays T, U of same length as time.
    Use definitions in prompt.
    """
    raise NotImplementedError

# Compute and plot energy
time_E = time  # reuse time vector
T, U = compute_energies(theta, dtheta, params)
E = T + U

plt.figure(figsize=(8, 4))
plt.plot(time_E, T, label='Kinetic (T)')
plt.plot(time_E, U, label='Potential (U)')
plt.plot(time_E, E, label='Total (E)')
plt.xlabel('Time (s)'); plt.ylabel('Energy'); plt.legend(); plt.title('Energy Terms'); plt.grid(True); plt.show()

# %% [markdown]
## Part B: Multisensory Integration (Assignment 2, Q7b)

Gaussian cue fusion without prior: measurement noise only.

Visual cue: $\mu_v=-10°$, $\sigma_v=5°$.  
Auditory cue: $\mu_a=25°$, $\sigma_a=15°$.

1. **Derive** (by hand) the MLE formula for combined estimate:
   $$\hat x = \frac{\sigma_a^2\mu_v + \sigma_v^2\mu_a}{\sigma_v^2 + \sigma_a^2}.$$  
   Show each algebraic step.

2. **Compute** \hat x in Python and print the result.  
   *Hint:* Use floating-point operations and verify with manual calculation.

3. **Plot** the likelihood functions $L_v(x)$ and $L_a(x)$ and their product (unnormalized posterior) over $x\in[-90,90]$.  
   *Hint:* Use `scipy.stats.norm.pdf` and normalize the product with `np.trapz`.

4. **Extended challenges:**
   a. Vary visual noise $\sigma_v\in[5,10,20,40]$ (keep $\sigma_a=15$), compute and plot \hat x vs $\sigma_v$.  
   b. Add a Gaussian prior $\mathcal{N}(0,\sigma_p^2)$ with $\sigma_p=20°$, derive and implement the MAP estimate. Compare numeric values.  
   c. Simulate $N=100$ trials: true source at 0°, sample noisy cues, compute MLE each trial, plot histogram of estimates, and compute MSE.  
   *Hint:* Use `np.random.seed` for reproducibility.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)
# Given values
mu_v, sigma_v = -10, 5
mu_a, sigma_a = 25, 15

# 2. Compute MLE estimate
x_hat = (sigma_a**2 * mu_v + sigma_v**2 * mu_a) / (sigma_v**2 + sigma_a**2)
print(f"MLE estimate: {x_hat:.2f}°")

# 3. Plot likelihoods & posterior
x = np.linspace(-90, 90, 1000)
Lv = norm.pdf(x, mu_v, sigma_v)
La = norm.pdf(x, mu_a, sigma_a)
post = Lv * La
post /= np.trapz(post, x)

plt.figure(figsize=(6, 4))
plt.plot(x, Lv, label='$L_v(x)$')
plt.plot(x, La, label='$L_a(x)$')
plt.plot(x, post, label='Posterior')
plt.xlabel('x (°)'); plt.legend(); plt.title('Likelihoods and Posterior'); plt.show()

# 4a. Vary sigma_v
sigmas = [5, 10, 20, 40]\ests = [(sigma_a**2 * mu_v + sv**2 * mu_a) / (sv**2 + sigma_a**2) for sv in sigmas]
plt.plot(sigmas, ests, 'o-'); plt.xlabel('$\sigma_v$'); plt.ylabel('$\hat x$'); plt.title('Estimates vs Visual Noise'); plt.show()

# 4b. MAP with prior sigma_p=20
sigma_p = 20
x_map = (mu_v/sigma_v**2 + mu_a/sigma_a**2) / (1/sigma_v**2 + 1/sigma_a**2 + 1/sigma_p**2)
print(f"MAP estimate: {x_map:.2f}°")

# 4c. Simulate trials
n_trials = 100
true_val = 0
ests = []
for _ in range(n_trials):
    v = np.random.normal(true_val, sigma_v)
    a = np.random.normal(true_val, sigma_a)
    est = (sigma_a**2 * v + sigma_v**2 * a) / (sigma_v**2 + sigma_a**2)
    ests.append(est)

plt.hist(ests, bins=20); plt.title('Histogram of MLEs'); plt.xlabel('Estimate (°)'); plt.ylabel('Count'); plt.show()

mse = np.mean((np.array(ests) - true_val)**2)
print(f"MSE over {n_trials} trials: {mse:.2f}")
