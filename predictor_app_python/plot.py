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



def plot_ecdf(path):
    # Generate plots
    results_df = pd.read_csv(f"{path}/results.csv", sep=",", header=0)
    print(f"[INFO] Generating plots for {path}")
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


    fig.savefig(f"{path}/ecdf.pdf")


def plot_timings(path):
    # Generate plots
    results_df = pd.read_csv(f"{path}/results.csv", sep=",", header=0)
    print(f"[INFO] Generating plots for {path}")
    
    results_df=results_df[(results_df["itime_latency"]<1000)&(results_df["itime_energy"]<1000)]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharex=True)

    ecdf = ECDF(results_df.itime_latency)
    x = np.unique(results_df.itime_latency)
    y = ecdf(x)
    ax.plot(x, y, c="red",label="Inference Time (Latency)")

    ecdf = ECDF(results_df.itime_energy)
    x = np.unique(results_df.itime_energy)
    y = ecdf(x)
    ax.plot(x, y, c="blue",label="Inference Time (Energy)")

    ax.grid(True, zorder=0)
    
    ax.legend()
    ax.set_xlabel("Inference Time [ns]", fontsize=16)
    ax.set_ylabel("ECDF", fontsize=16)


    fig.savefig(f"{path}/timings.pdf")


plot_ecdf("results/python/")
plot_ecdf("results/cplus/")


plot_timings("results/python/")
plot_timings("results/cplus/")