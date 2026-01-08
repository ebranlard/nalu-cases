import numpy as np
import os
import matplotlib.pyplot as plt
from welib.essentials import *
from welib.weio.csv_file import CSVFile
from welib.weio.fast_output_file import FASTOutputFile
from helper_functions import load_json_chirp
from helper_functions import analyse_step_with_tStart
from helper_functions import analyse_step, load_ULS
from welib.essentials import *
from welib.tools.figure import setFigureTitle

setFigurePath('../nalu_torque_2026/figs/')
setFigureFont(12)
setFigureTitle(1)
export=False
export=True


# --------------------------------------------------------------------------------}
# ---  
# --------------------------------------------------------------------------------{
IPlot =[]
IPlot +=[0] # Fidelity comparison
IPlot +=[1] # Comparison with theory

# --------------------------------------------------------------------------------}
# ---  Helpers
# --------------------------------------------------------------------------------{
chord = 1
span  = 4

pWag = [0.165,  0.045, 0.335,  0.300 ] # Wagner / Jones   Pitch change
pKus = [0.500,  0.13 , 0.500,  1.000 ] # Kussner          Transverse Gust
pOF  = [0.3  ,  0.14 , 0.7  ,   0.53 ] # OpenFAST

s_th = np.linspace(0, 100, 500)
PhiKussner = 1 - pKus[0]*np.exp(-pKus[1] * s_th) - pKus[2]*np.exp(-pKus[3]*s_th) # Kussner~ with A1+A2=1
PhiWagner  = 1 - pWag[0]*np.exp(-pWag[1] * s_th) - pWag[2]*np.exp(-pWag[3]*s_th) # Waggner
PhiOF      = 1 - pOF [0]*np.exp(-pOF [1] * s_th) - pOF [2]*np.exp(-pOF [3]*s_th) # OpenFAST



# --------------------------------------------------------------------------------}
# --- MAIN SCRIPTS
# --------------------------------------------------------------------------------{

cases=[]
cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':''    }]
cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HR' }]
cases+=[{'airfoil_name':'du00-w-212' , 'n':4  , 're':3   , 'suffix':''    }]
cases+=[{'airfoil_name':'du00-w-212' , 'n':22 , 're':3   , 'suffix':''    }]
cases+=[{'airfoil_name':'nlf1-0416'  , 'n':24 , 're':4   , 'suffix':''    }]
cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':''    }]
cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':'_HR' }]
# 




def aggregate(d0, d1):
    s_step0, phi_step0, s, phi = d0
    if d1 is None:
        d1 = {}
        d1['s_step']        = s_step0
        d1['s']             = s
        d1['phi_step']      = np.zeros((1, len(s_step0)))
        d1['phi_step'][0,:] = phi_step0
        d1['phi']           = np.zeros((1, len(s)))
        d1['phi'][0,:]      = phi
        print(d1['phi'].shape)
    else:
        phi_step2 =  np.interp(d1['s_step'], s_step0, phi_step0)
        phi2      =  np.interp(d1['s'     ], s      , phi)
        d1['phi_step'] = np.vstack( (d1['phi_step'], phi_step2))
        d1['phi']      = np.vstack( (d1['phi'], phi2))
        print(d1['phi'].shape)
    return d1

def plotDatAgg(ax, dat, label=None, color='k', lw=1.5, ls='-', alpha=0.3):
    s        = dat['s']
    s_step   = dat['s_step']
    phimin   = dat['phi'].min(axis  = 0)
    phimax   = dat['phi'].max(axis  = 0)
    phimean  = dat['phi'].mean(axis = 0)
    ax.fill_between(s, phimin, phimax, alpha=0.1, color=color)
    phimin   = dat['phi_step'].min(axis  = 0)
    phimax   = dat['phi_step'].max(axis  = 0)
    phimean  = dat['phi_step'].mean(axis = 0)
    ax.fill_between(s_step, phimin, phimax, alpha=alpha, color=color)
    ax.plot(s_step, phimean, c=color, label=label, lw=lw, ls=ls)


# --------------------------------------------------------------------------------}
# --- Agregate ot CFD, UA OF
# --------------------------------------------------------------------------------{
# --- ULS
cs = cases[0]
yml_path  = '_results/cases_chirp_n{}/{}/{}_re{:04.1f}_mean00_A01{:s}.yaml'.format(cs['n'], cs['airfoil_name'], cs['airfoil_name'], cs['re'], cs['suffix'])
json_path = yml_path.replace('.yaml','.json')
uls_outb  = yml_path.replace('.yaml', '_ULS_OmegaM.csv')
cfd_outb  = yml_path.replace('.yaml', '_CFD.outb')
dfc = FASTOutputFile(cfd_outb).toDataFrame()
dfl = load_ULS(uls_outb, dfc)
info, dfm = load_json_chirp(json_path, verbose = False, plot = False)
p,fig, datULS = analyse_step(dfl, info, plot=False, label='ULS', c=fColrs(2), fig=None, doFit=False)
datULS = aggregate(datULS, None)
fig=None
datCFD=None
datUA={2:None, 3:None, 4:None, 5:None}
UAMods = [2, 3, 5]
for ic, cs in enumerate(cases):
# for cs in [cases[0]]:
    base = cs['airfoil_name'] + '_re{:05.2f}M'.format(cs['re']) + cs['suffix']
    print(f"------------------------- {base}------------- {cs['suffix']}")
    yml_path  = '_results/cases_chirp_n{}/{}/{}_re{:04.1f}_mean00_A01{:s}.yaml'.format(cs['n'], cs['airfoil_name'], cs['airfoil_name'], cs['re'], cs['suffix'])
    json_path = yml_path.replace('.yaml','.json')
    dvr_path  = yml_path.replace('.yaml', '_UAA.dvr')
    cfd_outb  = yml_path.replace('.yaml', '_CFD.outb')
    uls_outb  = yml_path.replace('.yaml', '_ULS_OmegaM.csv')
    # --- JSON Info
    info, dfm = load_json_chirp(json_path, verbose = False, plot = False)

    dfc = FASTOutputFile(cfd_outb).toDataFrame()

    p,fig, datCFD0 = analyse_step(dfc, info, plot=False, label='CFD' if ic==0 else None, c=fColrs(1), fig=None, doFit=False)
    datCFD = aggregate(datCFD0, datCFD)
#     #for im, UAMod in enumerate([2, 3, 4, 5]):
    for im, UAMod in enumerate(UAMods):
#         for ip, (p,lab) in enumerate(zip([pWag, pKus, pOF] , ['Wg', 'Ks', 'OF'])):
        #for ip, (p,lab) in enumerate(zip([pWag, pKus, pOF] , ['Wg', 'Ks', 'OF'])):
#         for ip, (p,lab) in enumerate(zip([pWag, pOF] , ['Wg', 'OF'])):
        for ip, (p,lab) in enumerate(zip([pOF] , ['OF'])):
            uaa_outb  = yml_path.replace('.yaml', '_UA{}_{}.outb'.format(UAMod, lab))
            dfu = FASTOutputFile(uaa_outb).toDataFrame()
            p,fig, dat = analyse_step(dfu, info, plot=False, label=None, fig=None, doFit=False, useCl=True)
            datUA[UAMod] = aggregate(dat, datUA[UAMod])

if 0 in IPlot:

    fig, ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,3.8))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.99, bottom=0.125, hspace=0.20, wspace=0.20)

    plotDatAgg(ax, datCFD  , label='CFD', color=fColrs(1))
    plotDatAgg(ax, datUA[2], label='UA2', color=fColrs(2), ls='--')
    plotDatAgg(ax, datUA[3], label='UA3', color=fColrs(3), ls='--')
    plotDatAgg(ax, datUA[5], label='UA5', color=fColrs(4), ls='--', lw=2)
    plotDatAgg(ax, datULS, label='ULS', color=(0.3,0.3,0.3), lw=2.5, ls=':')

    ax.set_ylabel(r"Normalized lift $\phi(s)$ [-]")
    ax.set_xlabel(r"Dimensionless time, $s=2 U t / c$ [-]")
    ax.set_xlim([-0.0, 20])
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.tick_params(direction='in', top=True, right=True, labelright=False, labeltop=False, which='both')
    fig._title='StepResponseMultiFidelity'


if 1 in IPlot:
    fig, ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,3.8))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.99, bottom=0.125, hspace=0.20, wspace=0.20)
    plotDatAgg(ax, datCFD, label='CFD', color=fColrs(1))
    plotDatAgg(ax, datUA[5], label='UA5', color=fColrs(4), lw=2, ls='--')
    ax.plot(s_th, PhiWagner , '-', label='(Wagner Function) '  , lw=2, c='k')
    ax.plot(s_th, PhiKussner, ':', label='(Kussner Function)'  , lw=2, c='k')
    ax.plot(s_th, PhiOF     , '--', label='(OpenFAST Function)', lw=2, c='k')
    ax.legend()
    ax.set_ylabel(r"Normalized lift $\phi(s)$ [-]")
    ax.set_xlabel(r"Dimensionless time, $s=2 U t / c$ [-]")
    ax.set_xlim([-0.0, 20])
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.tick_params(direction='in', top=True, right=True, labelright=False, labeltop=False, which='both')
    fig._title='StepResponseTheory'


export2pdf()
plt.show()






