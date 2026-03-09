
""" 
Plot the chirp signal, with an example of Cl with UA and CFD

NOTE: s_factor = (2 * U) / chord   

s = t * s_factor = t * 2 U /chord

"""
import matplotlib.pyplot as plt
import numpy as np

import welib.weio as weio
from welib.weio.fast_output_file import FASTOutputFile
from welib.essentials import *
from welib.tools.figure import setFigureTitle

from helper_functions import load_json_chirp, plot_chirp_full_time, load_ULS
from helper_functions import split_chirp, postpro_chirp_tf, postpro_cycles_tf
from helper_functions import *


setFigurePath('../nalu_torque_2026/figs/')
setFigureFont(12)
# setFigureTitle(1)
export=True
# export=False

HR=False
HR=True


if HR:
    json_path = '_results/_data_paper/overview/S809_re00.8_mean00_A01_HRCAT.json'
    cfd_outb  = '_results/_data_paper/overview/S809_re00.8_mean00_A01_HRCAT_CFD.outb'
    ua2_outb  = '_results/_data_paper/overview/S809_re00.8_mean00_A01_HRCAT_UA2_OF.outb'
    ua5_outb  = '_results\_data_paper\overview\S809_re00.8_mean00_A01_HRCAT_UA5_OF.outb'
    uls_csv   = '_results\_data_paper\overview\S809_re00.8_mean00_A01_HRCAT_ULS.csv'
else:
    json_path = '_results/_data_paper/overview/S809_re00.8_mean00_A01.json'
    cfd_outb  = '_results/_data_paper/overview/S809_re00.8_mean00_A01_CFD.outb'
    ua2_outb  = '_results/_data_paper/overview/S809_re00.8_mean00_A01_UA2_OF.outb'
    ua5_outb  = '_results\_data_paper\overview\S809_re00.8_mean00_A01_UA5_OF.outb'
    uls_csv   = '_results\_data_paper\overview\S809_re00.8_mean00_A01_ULS_OmegaM.csv'

# --- JSON Info
info, dfm    = load_json_chirp(json_path, verbose = False, plot = False)

# --- Read postprocessed CFD (see t41_chirp_ua)
dfc = FASTOutputFile(cfd_outb).toDataFrame()
df2= FASTOutputFile(ua2_outb).toDataFrame()
df5= FASTOutputFile(ua5_outb).toDataFrame()
dfl = load_ULS(uls_csv, dfc)

# --- Pretty plot of chirp
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> REMEMBER I FLIPPED THE SIGN OF PITCH FOR FIGURE')

# --- JSON Info
cs={'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HRCAT' , 'Cl_alpha':6.50842, 'alpha0':  -1.00410  }
info.update(cs)


# # --- Remove Cl0
t_ref = (info['indices_phases'][0]-10)*info['dt'] # Time at end of transients
print('>>> t_ref', t_ref)
it = np.argmin(np.abs(dfc['Time_[s]'].values - t_ref))
dfc_ref = dfc['Cl_[-]'].values[it]
# df0_ref = df0['Cl_[-]'].values[it]
# df2_ref = df2['Cl_[-]'].values[it]
df5_ref = df5['Cl_[-]'].values[it]
dfl_ref = dfl['Cl_[-]'].values[it]
print('Ref values: ', dfc_ref, df5_ref, dfl_ref)
dfc['Cl_[-]']  -= dfc_ref
# df0['Cl_[-]']  -= df0_ref
# df2['Cl_[-]']  -= df2_ref
df5['Cl_[-]']  -= df5_ref
dfl ['Cl_[-]'] -= dfl_ref


_, trc, stc, chc, dwc = split_chirp(info, dfm, dfc, plot=False)
_, tru, stu, chu, dwu = split_chirp(info, dfm, df5, plot=False)
_, trl, stl, chl, dwl = split_chirp(info, dfm, dfl, plot=False)


dfc ['Time_[s]'] *= info['s_factor']
df2['Time_[s]'] *= info['s_factor']
df5['Time_[s]'] *= info['s_factor']
dfl ['Time_[s]'] *= info['s_factor']
dfm ['Time_[s]'] *= info['s_factor']

# it = np.argmin(np.abs(dfc['Time_[s]'].values - 7.7))
# cl_c  = dfc['Cl_[-]'].values[it]
# cl_2 = df2['Cl_[-]'].values[it]
# cl_5 = df5['Cl_[-]'].values[it]
# cl_l  = dfl ['Cl_[-]'].values[it]
# df5['Cl_[-]'] = df5['Cl_[-]'] - cl_5+cl_c
# dfl ['Cl_[-]'] = dfl ['Cl_[-]'] - cl_l +cl_c


for zoom in [0,1]:

    if HR:
        dfm['angle_[deg]'] *=-1 # <<<< TODO 
    else:
        dfm['angle_[deg]'] *=-1 # <<<< TODO 
    figsize=(12.8,4.8)
    fig = plot_chirp_full_time(info, dfm, dff=dfc, label='CFD', col=fColrs(1), dimLessTime=True, HR=HR, figsize=figsize)



    ax = fig.axes[0]
    ax.set_ylim([-1.4, 1.4])
    # ax.set_xscale('log')
    ax = fig.axes[1]
    ax.plot(df5['Time_[s]'], df5['Cl_[-]'], label='UA5', c=fColrs(4), ls='--', lw=1.0)
    ax.plot( dfl['Time_[s]'], dfl['Cl_[-]'] , label='ULS', c=(0.3, 0.3, 0.3), ls='-.', lw=1.0)
    ax.legend()
    fig.subplots_adjust(left=0.058, right=0.998, top=0.995, bottom=0.10, hspace=0.05, wspace=0.20)
    ax.set_ylim([-0.17, 0.17])
    ax.set_ylabel(r'$C_l-C_{l,0}$ [-]')
    ax.set_xlabel(r'Dimensionless time, $s=2Ut/c$ [-]')
    # #     ax2.set_xlabel(r'Dimensionless time, $2Ut/c$ [-]')
    if HR:
        fig._title = 'CombinedSimOverviewHR'
    else:
        fig._title = 'CombinedSimOverview'
    if zoom:
        fig._title += 'Zoom'
        ax.set_xlim([2426, 2845])
        ax.legend(loc='lower left', ncol=3)



# 
if export:
    export2pdf()

plt.show()
