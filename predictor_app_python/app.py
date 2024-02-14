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
import os, time
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from lpu_models import LPUModels

# Load ground truth
print("[INFO] Loading required files ...")
cpu_ground_truth_df = pd.read_csv("data/CPU_dataset.csv", sep=",", header=0)
gpu_ground_truth_df = pd.read_csv("data/GPU_dataset.csv", sep=",", header=0)

# Results storage file
f = open("results/results.csv", "w")
f.write("SNR,MCS,PRBs,TBS,m_type,predicted_dec_time,dec_time,p_err,itime_latency,itime_energy\n")

def estimate_service_time(SNR, MCS, PRBs, TBS):
    # Run LPU Models
    t_cpu, e_cpu, t_gpu, e_gpu,times = lpu_models.estimate_service_time(SNR, MCS, PRBs, TBS)

    # Compute prediction error
    cpu_tmp_df = cpu_ground_truth_df[(cpu_ground_truth_df.SNR_dB == SNR) & (cpu_ground_truth_df.MCS == MCS) & (cpu_ground_truth_df.TBS == TBS) & (cpu_ground_truth_df.nPRB == PRBs)]
    for cpu_val in cpu_tmp_df.Total_latency_us.values:
        err = 100*((t_cpu-cpu_val)/cpu_val)
        # Dump results into file
        f.write(f"{SNR},{MCS},{PRBs},{TBS},CPU,{t_cpu},{cpu_val},{err},{times[1]},{times[0]}\n")
    
    gpu_tmp_df = gpu_ground_truth_df[(gpu_ground_truth_df.SNR_dB == SNR) & (gpu_ground_truth_df.MCS == MCS) & (gpu_ground_truth_df.TBS == TBS) & (gpu_ground_truth_df.nPRB == PRBs)]
    for gpu_val in gpu_tmp_df.Total_latency_us.values:
        err = 100*((t_gpu-gpu_val)/gpu_val)
        # Dump results into file
        f.write(f"{SNR},{MCS},{PRBs},{TBS},GPU,{t_gpu},{gpu_val},{err},{times[3]},{times[2]}\n")

# Load Trace
input_file = "data/traces_236.8.csv"
trace_df = pd.read_csv(input_file, sep=" ", header=0)

# Create an instance of LPU Models
lpu_models = LPUModels()

# Run Inference
print("[INFO] Running Inference")
combinations = trace_df.groupby(['SNR', 'MCS', 'PRBs']).size()
for i, v in combinations.items():
    tbs = np.unique(trace_df[(trace_df.SNR == i[0]) & (trace_df.MCS == i[1]) & (trace_df.PRBs == i[2])].TBS.values)[0]
    estimate_service_time(i[0], i[1], i[2], tbs)
f.close()

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
print("[INFO] Results generated in the predictor_app_python/results folder")