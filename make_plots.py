import pandas as pd
from matplotlib import axes, pyplot as plt
import numpy as np
import seaborn as sns
# Reads the results to a pandas dataframe
df = pd.read_csv("results.csv")
# Deletes rows where both LP and RMP do not return a solution.
df = df.loc[~((df['MC Probability LP'] == 0) & (df['MC Probability RMP'] == 0)),:]
# Plots scatter plot of probability versus percentage improvement
fig = plt.figure()
plt.scatter(df["MC Probability LP"], df["Percentage Improvement"], marker="x")
plt.yscale('log')
plt.savefig("Scatter.png")
plt.ylabel("")
# Plots scatter plot of size versus runtime grouped by correlation size.
fig, ax = plt.subplots()
sizes = df["Correlation Size"].sort_values(ascending=False).unique()
colors = ["g", "r", "b", "c", "m"]
styles = [".", "o", "x", "+", "v"]
for i in range(len(colors)):
    curr = df.loc[(df["Correlation Size"] == sizes[i]),:]
    curr.plot(x="MC Probability LP", y="Runtime Delta", kind="scatter", ax=ax, color=colors[i], label=sizes[i], marker=styles[i])
ax.set_yscale('log')
ax.legend()
plt.savefig("runtime")