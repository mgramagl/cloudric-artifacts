'''
    @author:
        - Leonardo Lo Schiavo
    @contributors
        - Gines Garcia-Aviles
        - Marco Gramaglia
        - Andres Garcia-Saavedra
    @affiliation:
        - IMDEA Networks institute
'''

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Generate plots
results_df = pd.read_csv("results/results.csv", sep=",", header=0)
print("[INFO] Generating results")
cpu_results_df = results_df[results_df.m_type == "CPU"]
gpu_results_df = results_df[results_df.m_type == "GPU"]

fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharex=True)

ecdf = ECDF(cpu_results_df.p_err)
x = np.unique(cpu_results_df.p_err)
y = ecdf(x)
ax.plot(x, y, c="red")

ecdf = ECDF(gpu_results_df.p_err)
x = np.unique(gpu_results_df.p_err)
y = ecdf(x)
ax.plot(x, y, c="blue")

ax.grid(True, zorder=0)
patch_1 = mpatches.Patch(color="red", label='CPU')
patch_2 = mpatches.Patch(color="blue", label='GPU')

ax.legend(handles=[patch_1, patch_2], fontsize=16)

ax.set_xlim(-40, 60)
ax.set_xlabel("Prediction Error (%)", fontsize=16)
ax.set_ylabel("ECDF", fontsize=16)


fig.savefig("results/ecdf.pdf")