# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:00:19 2021

@author: Owner
"""

# =============================================================================
# packages
# =============================================================================
import math
import pandas as pd
import numpy as np
from statistics import NormalDist
import random
import numpy
import matplotlib.pyplot as plt
# =============================================================================
# a. Use this approach to solve again the exercise at the end of the lecture, but
# this time using a grid approximation. All numbers are in Newtons:
# Your prior about the box is
# 0  = 3.5
# with confidence
# 0  =1
# Your confidence in your perception is
# 1  x
# =
# Your lift the box multiple times and perceive the weight to be 6, 6, 7, 7, 4
# and 5.
# Plot the posterior of distribution of the weight after each lift on a grid of 500
# points between 0 and 10
# =============================================================================
m0 = 3.5
r = [4,4,4]
sigma0 = 1
sigma1 = 1
X_lift = [6,6,7,7,4,5]
Observations = []
for i in range(500):
    x = random.randint(0, 10) 
    Observations.append(x)
    
def postprior(x):
     postprior_arr = []
     s = (1/(math.sqrt(2*math.pi)*sigma1))
     '''lik1 = math.exp(pow(-(x - Observations[1]),2)/2)
     prior = s*math.exp(pow(-(x - m0),2)/2)
     postprior_arr.append(s*lik1*prior)'''
     for i in range(500):
         A = math.exp((-1)*pow((x - Observations[i]),2)/2)
         B = math.exp((-1)*pow((Observations[i] - m0),2)/2)
         postprior_arr.append(s*A*s*B)
     postprior_arr = numpy.array(postprior_arr)
     postprior = postprior_arr/ sum(postprior_arr)
     return postprior
x1 = postprior(X_lift[1])
df = pd.DataFrame(columns = ['X','Y'])
df['X'] = Observations
df['Y'] = x1
df.plot.kde()

plt.plot(df, np.full_like(df, -0.1), '|k', markeredgewidth=1)
plt.xlabel("Observations")
plt.ylabel("postprior")
plt.show()


