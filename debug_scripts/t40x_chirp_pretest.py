import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import chirp, butter, filtfilt
from scipy.optimize import curve_fit
from ua import compute_bl_response, get_analytical_tf
from system_dynamics import tfestimate, tfestimate_stitched
from helper_functions import generate_step_chirp


# --- 1. Configuration & Parameters ---
alpha_mean_deg=0.0
alpha_amp_deg=1.0

# Airfoil / Flow Properties (Wind Energy Context)
chord = 1.0  # m
U = 34.1  # m/s
C_L_ALPHA = 6.1  # 1/rad (NREL S809 approx)
MACH = 0.00 # Mach number (Low speed)
# B-L Attached Flow Constants (States 1 & 2)

A1, A2 = 0.165, 0.335
B1, B2 = 0.0455, 0.3


nT_steady       = 60   # TODO 10, 50 60
K_TARGET        = 1.0  # TODO 0.6 or 1
F0_FACTOR       = 4    # TODO 2 or 4, 5
N_CYCLES_DWELLS = 5    # TODO 5
N_CONV          = 600  # TODO 250, 400
N_CYCLES_CHIRP  = 4    # TODO 2

# nT_steady       = 50  # TODO 10 then 50
# K_TARGET        = 0.6 # TODO 0.6 or 1
# F0_FACTOR       = 2   # TODO 2 or 5
# N_CYCLES_DWELLS = 5   # TODO 5
# N_CONV          = 250  # TODO 10 then 250
# N_CYCLES_CHIRP  = 2   # TODO 2


DT_FACT         = 0.05 # TODO 0.02



dt = float(np.around(DT_FACT * chord / U, 8))

time, alpha_in, info= generate_step_chirp(dt, U, B1, alpha_mean_deg=alpha_mean_deg, alpha_amp_deg=alpha_amp_deg, 
                        n_chord_transient=nT_steady, n_chord_step=nT_steady, f0_factor=F0_FACTOR, k_target=K_TARGET, 
                        k_dwells=[0.1, 0.3, 0.5, 1],
                        n_cycles_dwells=N_CYCLES_DWELLS,
                        n_conv=N_CONV,
                        n_cycles_chirp=N_CYCLES_CHIRP,
                        flip=True,
                        chord=chord, verbose=True, plot=True)

dt=time[1]-time[0]
fs=1/dt

cl_total = compute_bl_response(time, alpha_in, U=U, Cl_alpha= C_L_ALPHA, A1=A1, A2=A2, B1=B2, MACH=MACH, chord=chord)

# Extract the indices from our info object
i_trans, i_step, i_chirp = info['indices_phases']

# The chirp starts after Transient + Step, and lasts for i_chirp steps
start_idx = i_trans + i_step
end_idx = start_idx + i_chirp

# Slicing the data for Transfer Function estimation
time_tf = time[start_idx:end_idx]
alpha_tf = alpha_in[start_idx:end_idx]
cl_tf    = cl_total[start_idx:end_idx]

# IMPORTANT: Subtract the mean to remove the DC offset before TF estimation
alpha_tf = alpha_tf - np.mean(alpha_tf)
cl_tf    = cl_tf - np.mean(cl_tf)


# Estimate Numerical Transfer Function using CPSD/PDS (ETFE)
# Using alpha_tf as input and cl_total as output
from scipy.signal.windows import tukey
# # Apply a window that preserves the low-frequency start
#  window = tukey(len(alpha_tf), alpha=0.1)
# f_num2, H_num2 = tfestimate(alpha_tf * window, cl_total * window, fs=1/dt)
f_num, H_num, f_l, H_l, f_h, H_h = tfestimate_stitched(alpha_tf, cl_tf, fs=1/dt, f_stitch=0.1*info['f1'], returnAll=True)
f_num2, H_num2 = tfestimate(alpha_tf, cl_tf, fs=1/dt)

# Get Analytical TF for comparison
H_th, H_circ, H_nc = get_analytical_tf(f_num, U=U, Cl_alpha=C_L_ALPHA, A1=A1, A2=A2, B1=B1, B2=B2, MACH=MACH, chord=chord)

# --- 6. Plotting ---
fig, axes = plt.subplots(3, 1, sharey=False, figsize=(12.8,5.8))
fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
# Time Domain Plot
ax = axes[0]
ax.plot(time, np.degrees(alpha_in), label='Alpha [deg]')
ax.plot(time, cl_total, label='Cl (B-L Model)')
ax.set_xlabel("Time [s]")
ax.legend()
ax.grid(True)

# Frequency Domain Plot (Bode Magnitude)
ax = axes[1]
ax.semilogx(f_num, 20 * np.log10(np.abs(H_num)), label='Num (stitched)', linewidth=2)
ax.semilogx(f_num2, 20 * np.log10(np.abs(H_num2)), label='Num ', linewidth=2)
ax.semilogx(f_num, 20 * np.log10(np.abs(H_th)), '--', label='Analytical', alpha=0.8)
# plt.semilogx(f_num, 20 * np.log10(np.abs(H_circ)), ':', label='Analytical - Circulatory', alpha=0.8)
# plt.semilogx(f_num, 20 * np.log10(np.abs(H_nc)), ':', label='Analytical - Non-Circulatory', alpha=0.8)
# plt.semilogx(f_l, 20 * np.log10(np.abs(H_l)), 'k:', label='low')
# plt.semilogx(f_h, 20 * np.log10(np.abs(H_h)), 'k:', label='high')
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Magnitude [dB]")
# plt.xlim(0.1, 20)
ax.legend()
ax.grid(True, which='both')

# Frequency Domain Plot (Bode Magnitude)
ax = axes[2]
ax.semilogx(f_num,  np.angle(H_num , deg=True), label='Numerical (from Chirp)', linewidth=2)
ax.semilogx(f_num2, np.angle(H_num2, deg=True), label='Numerical (from Chirp)', linewidth=2)
ax.semilogx(f_num,  np.angle(H_th  , deg=True), '--', label='Analytical', alpha=0.8)

axes[1].set_xlim(info['f0'], info['f1'])
axes[2].set_xlim(info['f0'], info['f1'])
print('info', info['f1'], info['f0'])
# plt.title("Bode Plot: Transfer Function Magnitude")
# plt.tight_layout()
plt.show()
