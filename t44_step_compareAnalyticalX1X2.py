import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from welib.weio.fast_output_file import FASTOutputFile
from welib.weio.csv_file import CSVFile

pWag = [0.165,  0.045, 0.335,  0.300 ] # Wagner / Jones   Pitch change
# pKus = [0.500,  0.13 , 0.500,  1.000 ] # Kussner          Transverse Gust
# pOF  = [0.3  ,  0.14 , 0.7  ,   0.53 ] # OpenFAST

# --- Numerical Inputs ---
Cl_alpha = 6.48300;
alpha0 = -1.0; 
tStep = 8.325; 
tStep = 8.33333  
tStep = 8.34166333

delta_alpha = -1.0  # [deg]
chord = 1.0; 
U = 6.0; 
# Tf0 = 6.0; Tp0 = 1.5





# --- Load Data (Assumed df already exists or loaded from csv) ---
# df = FASTOutputFile('./_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_UA5_OF.outb').toDataFrame()
# A1, b1, A2, b2 = pOF # <<<<<<<<<<<<<<<<<<<<<
# df = FASTOutputFile('./_results/_data_paper/DebugRate_KillX3_SimMod2/S809_re00.8_mean00_A01_UA5_Wg_SimMod2_NoRate.outb').toDataFrame()
df = FASTOutputFile('./_results/_data_paper/DebugRate_KillX3_SimMod2/S809_re00.8_mean00_A01_UA5_Wg_SimMod2.outb').toDataFrame()
df = FASTOutputFile('./_results/_data_paper/DebugRate_KillX3_SimMod2/S809_re00.8_mean00_A01_UA5_Wg_SimMod2_NoRate.outb').toDataFrame()
A1, b1, A2, b2 = pWag # <<<<<<<<<<<<<<<<<<<<<
A1, b1, A2, b2 = pWag # <<<<<<<<<<<<<<<<<<<<<

# --- Analytical Calculation ---
t = df['Time_[s]'].values
Tu = 0.5 * chord / U
print('>>>>>>>>>>Tu', Tu, df['Tu_[s]'].mean())
# Heaviside-like mask for the step starting at tStep
mask = (t >= tStep)
t_rel = (t - tStep) * mask # Time relative to step start

# States x1 and x2 (converted to degrees for alphaE consistency, or kept in rad if alpha_34 is rad)
# Assuming delta_alpha is provided in degrees to match alphaE_[deg]
x1_ana = delta_alpha * A1 * (1 - np.exp(-b1 * t_rel / Tu)) * mask
x2_ana = delta_alpha * A2 * (1 - np.exp(-b2 * t_rel / Tu)) * mask

# Effective Alpha (Eq 13): alpha_E = alpha_34*(1-A1-A2) + x1 + x2
# alpha_34 is delta_alpha after tStep, 0 before.
alpha34 = delta_alpha * mask
alphaE_ana = alpha34 * (1 - A1 - A2) + x1_ana + x2_ana

# Lift Coefficient (Eq 14, neglecting omega): Cl = Cl_alpha * (alphaE - alpha0)
cl_circ = Cl_alpha * np.deg2rad(alphaE_ana - alpha0)

# ---
df_inp = CSVFile('./_results/_data_paper/DebugRate_KillX3_SimMod2/AeroTSFile.csv').toDataFrame()
print(df_inp.keys())
t_pr = df_inp['Time_[s]'].values
pr = df_inp['Pitch_rate_[rad/s]'].values
cl_pr = np.interp(t, t_pr, pr) * np.pi * Tu # pi Tu Omega

cl_ana= cl_circ + cl_pr

# --- Comparison Plotting ---
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axes[0].plot(t, df['x1_[rad]'], 'k-', label='Data x1'); 
axes[0].plot(t, np.deg2rad(x1_ana), 'r--', label='Ana x1'); 
axes[1].plot(t, df['x2_[rad]'], 'k-', label='Data x2'); 
axes[1].plot(t, np.deg2rad(x2_ana), 'r--', label='Ana x2'); 
axes[2].plot(t, df['alphaE_[deg]'], 'k-', label='Data alphaE'); 
axes[2].plot(t, alphaE_ana, 'g--', label='Ana alphaE'); 
axes[3].plot(t, df['Cl_[-]'], 'k-', label='Data Cl'); 
axes[3].plot(t, cl_ana, 'b--', label='Ana Cl'); 
axes[3].plot(t, cl_circ, 'g--', label='Ana Cl (circ only)'); 

axes[0].set_ylabel('x1 [rad]')
axes[1].set_ylabel('x2 [rad]')
axes[2].set_ylabel('alphaE [deg]')
axes[3].set_ylabel('Cl [-]')
for ax in axes: 
    ax.legend(loc='right'); 
    ax.grid(True, ls=':')
    ax.set_xlim([tStep-1,10])
axes[-1].set_xlabel('Time [s]'); fig.suptitle('Analytical vs CFD-Solver Comparison')
plt.show()
