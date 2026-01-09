
""" 
Plot the chirp signal, with an example of Cl with UA and CFD

NOTE: s_factor = (2 * U) / chord   

s = t * s_factor = t * 2 U /chord

"""

import matplotlib.pyplot as plt
import numpy as np
from welib.weio.fast_output_file import FASTOutputFile

from helper_functions import load_json_chirp, plot_chirp_full_time, load_ULS

from welib.essentials import *
from welib.tools.figure import setFigureTitle

setFigurePath('../nalu_torque_2026/figs/')
setFigureFont(12)
# setFigureTitle(1)
export=True

json_path = '_results/_data_paper/overview/S809_re00.8_mean00_A01.json'
cfd_outb  = '_results/_data_paper/overview/S809_re00.8_mean00_A01_CFD.outb'
uaa_outb  = '_results/_data_paper/overview/S809_re00.8_mean00_A01_UAA.outb'
ua2_outb  = '_results/_data_paper/overview/S809_re00.8_mean00_A01_UA2_OF.outb'
ua5_outb  = '_results\_data_paper\overview\S809_re00.8_mean00_A01_UA5_OF.outb'
uls_csv   = '_results\_data_paper\overview\S809_re00.8_mean00_A01_ULS_OmegaM.csv'

# --- JSON Info
info, dfm    = load_json_chirp(json_path, verbose = False, plot = False)

# --- Read postprocessed CFD (see t41_chirp_ua)
dfc = FASTOutputFile(cfd_outb).toDataFrame()
dfu = FASTOutputFile(uaa_outb).toDataFrame()
dfu2= FASTOutputFile(ua2_outb).toDataFrame()
dfu5= FASTOutputFile(ua5_outb).toDataFrame()
dfl = load_ULS(uls_csv, dfc)

# --- Pretty plot of chirp
fig = plot_chirp_full_time(info, dfm, dff=dfc, label='CFD', col=fColrs(1))

it = np.argmin(np.abs(dfc['Time_[s]'].values - 7.7))
cl_c  = dfc['Cl_[-]'].values[it]
cl_u  = dfu ['Cl_[-]'].values[it]
cl_u2 = dfu2['Cl_[-]'].values[it]
cl_u5 = dfu5['Cl_[-]'].values[it]
cl_l  = dfl ['Cl_[-]'].values[it]

ax = fig.axes[1]
# ax.plot(dfu ['Time_[s]'], dfu ['Cl_[-]'] - cl_u +cl_c, label='UA2 Old', c=fColrs(3), ls='--', lw=2)
# ax.plot(dfu2['Time_[s]'], dfu2['Cl_[-]'] - cl_u2+cl_c, label='UA2', c=fColrs(2), ls=':', lw=2)
ax.plot(dfu5['Time_[s]'], dfu5['Cl_[-]'] - cl_u5+cl_c, label='UA5', c=fColrs(4), ls=':')
ax.plot(dfl['Time_[s]'] , dfl ['Cl_[-]'] - cl_l +cl_c, label='ULS', c=(0.3, 0.3, 0.3), ls=':', lw=1.6)
ax.legend()
fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.09, hspace=0.20, wspace=0.20)
ax.set_ylim([0, 0.25])
ax.set_ylabel(r'$C_l$ [-]')
fig._title = 'CombinedSimOverview'


if export:
    export2pdf()

plt.show()
