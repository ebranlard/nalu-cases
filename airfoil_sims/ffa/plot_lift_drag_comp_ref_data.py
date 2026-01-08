# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl 
from cycler import cycler

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
print(colors)


mpl.rcParams['lines.linewidth'] = 2 
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 8
#mpl.rcParams['figure.figsize'] = (6.328, 5.328)
mpl.rcParams['figure.figsize'] = (7.328, 5.328)
mpl.rcParams["figure.autolayout"] = True

rho = 1.2
u_infty = 75.0
dyn_pres = 0.5 * rho * (u_infty ** 2)
N = 8000


#aoa_array = [-180, -175, -170, -165,
#                -160, -152, -144, -136, -128,
#                -120, -110, -100, 
#                -90, -80, -70, -60, -50, -40
#                -33, -30, -27, -24, -21, -18,
#                -15, -12, -9, -6, -3,
#                0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33,
#                40, 50, 60, 70, 80, 90,
#                100, 110, 120,
#                128, 136, 144, 152, 160,
#                165, 170, 175, 180]


# FFA_W3_211:
#af_name='ffa_w3_211'
#aoa = [-180, -175, -170, -165,
#           -160, -152, -144, -136, -128,
#           -120, -110, -100,
#           -90, -80, -70, -60, -50, -40,
#           -33, -30, -27, -24, -21, -18,
#           -15, -12, -9, -6, -3, 
#           0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 
#           40, 50, 60, 70, 80, 90, 100, 110, 120,
#           128, 136, 144, 152, 160,
#           165, 170, 175, 180]

#af_name='ffa_w3_241'
#aoa = [-180, -175, -170, -165, -160, -152, -144, -136, -128,
#       -120, -110, -100, -90, -80, -70, -60, -50, -40, 
#       -33, -30, -27, -24, -21, -18,
#       -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 
#       40, 50, 60, 70, 80, 90, 100, 110, 120,
#       128, 136, 144, 152, 160, 165, 170, 175, 180]

#af_name='ffa_w3_270'
#aoa = [-180, -175, -170, -165,
#                -160, -152, -144, -136, -128,
#                -120, -110, -100, 
#                -90, -80, -70, -60, -50, -40, -33,
#                -30, -27, -24, -21, -18,
#                -15, -12, -9, -6, -3, 
#                0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 
#                40, 50, 60, 70, 80, 90, 
#                100, 110, 120,
#                128, 136, 144, 152, 160,
#                165, 170, 175, 180]

#af_name='ffa_w3_301'
#aoa = [-180, -175, -170, -165, -160, -152, -144, -136, -128, 
#-120, -110, -100, -90, -80, -70, -60, -50, -40, -33, -30, -27, -24, -21, -18,
#-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 
#40, 50, 60, 70, 80, 90, 100, 110, 120,
#128, 136, 144, 152, 160, 165, 170, 175, 180]


#af_name='ffa_w3_330'
#aoa = [-180, -175, -170, -165, -160, -152, -144, -136, -128, -120, -110, -100,
#       -90, -80, -70, -60, -50, -40,
#       -33, -30, -27, -21, -18,
#       -15, -12, -3, 0, 12, 15, 18, 21, 24, 27, 30, 33, 40, 50, 60, 70, 80, 90, 100, 110, 120,
#       128, 136, 144, 152, 160,
#       165, 170, 175, 180]

# FFA_W3_360:
#af_name='ffa_w3_360'
#aoa = [-180, -175, -170, -165,
#           -160, -152, -144, -136, -128,
#           -120, -110, -100,
#           -90, -80, -70, -60, -50, -40,
#           -33, -30, -27, -24, -21, -18,
#           -15, -12, -9, -6, -3,
#           0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33,
#           40, 50, 60, 70, 80, 90,
#           100, 110, 120,
#           128, 136, 144, 152, 160,
#           165, 170, 175, 180]

# FFA_W3_500:
af_name='ffa_w3_500'
aoa = [-180, -175, -170, -165, -160,
-152, -144, -136, -128, -120, -110,
-100, -90, -80, -70, -60, -50, -40, -33, -30,
-27, -24, -21, -18, -15, -12,
-9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 40, 50, 60, 70, 80, 90,
100, 110, 120, 128,
136, 144, 152, 160,
165, 170, 175, 180]


#ref_data = {
#    'cl' : pd.read_csv('ref_data/cl_ffa_w3_211_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cl']),
#    'cd' : pd.read_csv('ref_data/cd_ffa_w3_211_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cd']),
#}

#ref_data = {
#    'cl' : pd.read_csv('ref_data/cl_ffa_w3_241_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cl']),
#    'cd' : pd.read_csv('ref_data/cd_ffa_w3_241_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cd']),
#}

#ref_data = {
#    'cl' : pd.read_csv('ref_data/cl_ffa_w3_270_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cl']),
#    'cd' : pd.read_csv('ref_data/cd_ffa_w3_270_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cd']),
#}

#ref_data = {
#    'cl' : pd.read_csv('ref_data/cl_ffa_w3_301_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cl']),
#    'cd' : pd.read_csv('ref_data/cd_ffa_w3_301_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cd']),
#}

#ref_data = {
#    'cl' : pd.read_csv('ref_data/cl_ffa_w3_330_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cl']),
#    'cd' : pd.read_csv('ref_data/cd_ffa_w3_330_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cd']),
#}

#ref_data = {
#    'cl' : pd.read_csv('ref_data/cl_ffa_w3_360_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cl']),
#    'cd' : pd.read_csv('ref_data/cd_ffa_w3_360_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cd']),
#}

ref_data = {
    'cl' : pd.read_csv('ref_data/cl_ffa_w3_500_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cl']),
    'cd' : pd.read_csv('ref_data/cd_ffa_w3_500_ref_data.txt',header=None,sep='\t', lineterminator='\n',names=['aoa','cd']),
}

def get_avg_data():
    """Get average lift and drag data for a given grid"""
    case_data = [pd.read_csv(af_name+'/'+'data_files/'+af_name+'_'+str(aoa[i])+'.dat', sep="\s+", skiprows=1,header=None, names=[ "Time","Fpx","Fpy","Fpz","Fvx","Fvy","Fvz","Mtx","Mty","Mtz","Y+min","Y+max"],dtype=float).iloc[-N:] for i, c in enumerate(aoa)]
    case_cl = [ np.average((c["Fpy"] + c["Fvy"])/dyn_pres/4.0) for c in case_data]
    case_cd = [ np.average((c["Fpx"] + c["Fvx"])/dyn_pres/4.0) for c in case_data]
    return case_cl, case_cd

def get_std_data():
    """Get average lift and drag data for a given turbulence model, upwinding and grid"""
    case_data = [pd.read_csv(c+'/'+'iddes_ffa_w3_211_'+str(aoa[i])+'.dat', sep="\s+", 
skiprows=1,header=None, names=[ "Time","Fpx","Fpy","Fpz","Fvx","Fvy","Fvz","Mtx","Mty","Mtz","Y+min","Y+max"],dtype=float).iloc[-N:] for i, c in enumerate(case_dirs)]
    case_cl_std = [ np.std((c["Fpy"] + c["Fvy"])/dyn_pres/4.0) for c in case_data]
    case_cd_std = [ np.std((c["Fpx"] + c["Fvx"])/dyn_pres/4.0) for c in case_data]
    return case_cl_std, case_cd_std

def get_case_data(i):
    return pd.read_csv(af_name+'/'+'data_files/'+af_name+'_'+str(aoa[i])+'.dat', sep="\s+", skiprows=1,header=None, names=[ "Time","Fpx","Fpy","Fpz","Fvx","Fvy","Fvz","Mtx","Mty","Mtz","Y+min","Y+max"],dtype=float)

with PdfPages('cl_'+af_name+'_time.pdf') as pfpgs:
     fig = plt.figure()
     case_cl, case_cd = get_avg_data()

     for i,c in enumerate(aoa):
         cdata = get_case_data(i)
         cl = (cdata["Fpy"] + cdata["Fvy"])/dyn_pres/4.0
         plt.plot(cdata["Time"].iloc[100:], cl.iloc[100:], label=aoa[i])
     plt.xlabel('Time (s)')
     plt.ylabel(r'$C_l$')
     plt.legend(loc='lower right', ncol=8)
     plt.tight_layout()
     pfpgs.savefig()
     plt.close(fig)

with PdfPages('cl_'+af_name+'_large_aoa.pdf') as pfpgs:
     fig = plt.figure()
     case_cl, case_cd = get_avg_data()

     for i,c in enumerate(aoa):
         if (aoa[i] == 144):
            cdata = get_case_data(i)
            cl = (cdata["Fpy"] + cdata["Fvy"])/dyn_pres/4.0
            plt.plot(cdata["Time"].iloc[100:], cl.iloc[100:], label=aoa[i])
     plt.xlabel('Time (s)')
     plt.ylabel(r'$C_l$')
     plt.legend(loc='lower right', ncol=8)
     plt.tight_layout()
     pfpgs.savefig()
     plt.close(fig)

with PdfPages('cl_'+af_name+'.pdf') as pfpgs:
    fig = plt.figure()
    case_cl, case_cd = get_avg_data()
#    case_cl_std, case_cd_std = get_std_data()
#    case_cl_stdp = np.add(case_cl,case_cl_std)
#    case_cl_stdm = np.subtract(case_cl,case_cl_std)
    plt.plot(aoa, case_cl, marker = 'o', markersize=8, color=colors[0], linestyle = 'None')
#    plt.plot(aoa, case_cl_stdp, marker = '_', markersize=8, color=colors[0], linestyle = 'None')
#    plt.plot(aoa, case_cl_stdm, marker = '_', markersize=8, color=colors[0], linestyle = 'None')
#    plt.vlines(aoa, case_cl_stdp, case_cl_stdm, color=colors[0], linestyle='solid')
    plt.plot(ref_data['cl']['aoa'], ref_data['cl']['cl'],'+-', label='Ref. Data', color='black')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('$C_l$')
    plt.xlim([-182.0,182.0])
    plt.ylim([-1.6,2.5]) 
    plt.minorticks_on()
    plt.tight_layout()
    pfpgs.savefig()
    plt.close(fig)
    plt.show()

with PdfPages('cd_'+af_name+'.pdf') as pfpgs:
    fig = plt.figure()
    case_cl, case_cd = get_avg_data()
#    case_cl_std, case_cd_std = get_std_data()
#    case_cd_stdp = np.add(case_cd,case_cd_std)
#    case_cd_stdm = np.subtract(case_cd,case_cd_std)
    plt.plot(aoa, case_cd, marker = 'o', markersize=8, color=colors[0], linestyle = 'None')
#    plt.plot(aoa, case_cd_stdp, marker = '_', markersize=8, color=colors[0], linestyle = 'None')
#    plt.plot(aoa, case_cd_stdm, marker = '_', markersize=8, color=colors[0], linestyle = 'None')
#    plt.vlines(aoa, case_cd_stdp, case_cd_stdm, color=colors[0], linestyle='solid') 
    plt.plot(ref_data['cd']['aoa'], ref_data['cd']['cd'],'+-', label='Ref. Data', color='black')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('$C_d$')
    plt.xlim([-182.0,182.0]) 
    plt.ylim([-0.1,2.0])  
    plt.minorticks_on()
    plt.tight_layout()
    pfpgs.savefig()
    plt.close(fig)
