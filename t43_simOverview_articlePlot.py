
""" 
Plot the chirp signal, with an example of Cl with UA and CFD

NOTE: s_factor = (2 * U) / chord   

s = t * s_factor = t * 2 U /chord

"""

import matplotlib.pyplot as plt
from welib.weio.fast_output_file import FASTOutputFile

from helper_functions import load_json_chirp, plot_chirp_full_time

from welib.essentials import *
from welib.tools.figure import setFigureTitle

setFigurePath('../nalu_torque_2026/figs/')
setFigureFont(12)
# setFigureTitle(1)

json_path = '_results/_data_paper/overview/S809_re00.8_mean00_A01.json'
cfd_outb  = '_results/_data_paper/overview/S809_re00.8_mean00_A01_CFD.outb'
uaa_outb  = '_results/_data_paper/overview/S809_re00.8_mean00_A01_UAA.outb'

# --- JSON Info
info, dfc    = load_json_chirp(json_path, verbose = False, plot = False)

# --- Read postprocessed CFD (see t41_chirp_ua)
dff = FASTOutputFile(cfd_outb).toDataFrame()
dfa = FASTOutputFile(uaa_outb).toDataFrame()

# --- Pretty plot of chirp
fig = plot_chirp_full_time(info, dfc, dfm=None, dff=dff, label='CFD', col=fColrs(1))

ax = fig.axes[1]
ax.plot(dfa['Time_[s]'], dfa['Cl_[-]'], label='UAA', c=fColrs(4), ls='--')
ax.legend()
fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.09, hspace=0.20, wspace=0.20)
ax.set_ylim([0, 0.25])
ax.set_ylabel(r'$C_l$ [-]')
fig._title = 'CombinedSimOverview'


# if export:
export2pdf()

plt.show()
