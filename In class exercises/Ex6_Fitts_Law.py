# %%
# 📘 Lecture 6 In-Class Assignment: Fitts' Law and Movement Time
# Course: Models of the Motor System (Spring 2025)
# Objective: Analyze mouse-based pointing data to evaluate Fitts' Law using your own and peer data

# %%
# 🧰 Step 1: Setup and Imports
# We import required libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# %%
# 📂 Step 2: Load your data
# Load the CSV file you downloaded from the Fitts' Law demo site.
# Replace the filename if needed.
# Task: Load the data and preview the first few rows.

# Your code here


# %%
# 🧹 Step 3: Preprocess the data
# Filter to include only successful trials (hit == True).
# Compute the Index of Difficulty: log2(D / W + 1)
# Use the columns 'distanceFromPrevious' as D and 'radius' as W.
# Task: Create a new column called 'ID_log2' and clean your dataset.

# Your code here


# %%
# 📊 Step 4: Plot Movement Time vs Index of Difficulty
# Choose one participant (e.g., yourself) and create a scatterplot of movement time vs ID_log2.
# Task: Subset the data for one username and plot the result.

# Your code here


# %%
# 📈 Step 5: Fit Fitts' Law to your data
# Fit a linear regression of the form MT = a + b * ID_log2
# Task: Fit and report the slope, intercept, and R² for your selected participant.

# Your code here


# %%
# 📋 Step 6: Fit and Compare Across Participants
# Task: For each participant in the dataset, fit the Fitts' Law model.
# Create a summary table of intercept (a), slope (b), and R² for each participant.

# Your code here


# %%
# 📈 Step 7: Visualize Group Differences
# Create bar plots showing the intercept and slope for each participant.
# Task: Make side-by-side bar plots using seaborn or matplotlib.

# Your code here


# %%
# 🧠 Step 8: Reflect and Answer
# Write your answers in comments.

# 1. What does the intercept (a) represent in this model?
# 2. What does the slope (b) reflect about task difficulty?
# 3. Why might intercepts or slopes differ between participants?
# 4. Does Fitts' Law seem to describe your data well?

