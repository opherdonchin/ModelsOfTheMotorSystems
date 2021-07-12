#%% Imports

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#%% Predator / prey model

a = 0.2
b = 0.2
c = 0.0
d = 0.002


def predator_prey(xy, t):
    dxy_dt = [a*xy[0]-b*xy[0]*xy[1], d*xy[0]*xy[1]-c*xy[1]]
    return dxy_dt


y0 = 1
x0 = 1.5
t = np.linspace(0, 40, 500)

xy = odeint(predator_prey, [x0, y0], t)
prey = xy[:,0]
predator = xy[:,1]

plt.figure()
plt.plot(t,prey, "+", label="Prey")
plt.plot(t,predator, "*", label="Predator")
plt.xlabel('time')
plt.ylabel('population')
plt.legend()
plt.show()
