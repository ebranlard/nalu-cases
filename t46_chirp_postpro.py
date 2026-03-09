import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

from welib.weio.fast_output_file import FASTOutputFile
from nalulib.weio.csv_file import CSVFile

from welib.essentials import *

from helper_functions import postpro_chirp_tf, load_json_chirp, split_chirp, plotmelog
from ua import get_analytical_tf


# --------------------------------------------------------------------------------}
# --- MAIN SCRIPTS
# --------------------------------------------------------------------------------{
# --- Main inputs
cases=[]
# cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':''      , 'Cl_alpha':6.50842, 'alpha0':-1.00410 }]
# cases+=[{'airfoil_name':'du00-w-212' , 'n':22 , 're':3   , 'suffix':''      , 'Cl_alpha':6.71256, 'alpha0':-2.35240}]
# cases+=[{'airfoil_name':'nlf1-0416'  , 'n':24 , 're':4   , 'suffix':''      , 'Cl_alpha':6.85040, 'alpha0':-3.93070}] # <<<< Problem in polar
# cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':''      , 'Cl_alpha':6.76063, 'alpha0':-2.78090}]

cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HRCAT', 'Cl_alpha':6.50842, 'alpha0':-1.00410  }] # NOTE: 0.8 or 0.75
# cases+=[{'airfoil_name':'du00-w-212' , 'n':22 , 're':3   , 'suffix':'_HRCAT', 'Cl_alpha':6.43284, 'alpha0':-2.35240}]
# cases+=[{'airfoil_name':'nlf1-0416'  , 'n':24 , 're':4   , 'suffix':'_HRCAT', 'Cl_alpha':6.56495, 'alpha0':-3.93070}]
cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':'_HRCAT', 'Cl_alpha':6.76063, 'alpha0':-2.78090}]



pWag = [0.165,  0.045, 0.335,  0.300 ] # Wagner / Jones   Pitch change
pKus = [0.500,  0.13 , 0.500,  1.000 ] # Kussner          Transverse Gust
pOF  = [0.3  ,  0.14 , 0.7  ,   0.53 ] # OpenFAST


fig = None


for cs in cases:
# for cs in [cases[0]]:
    base = cs['airfoil_name'] + '_re{:05.2f}M'.format(cs['re']) + cs['suffix']
    print(f"------------------------- {base}------------- {cs['suffix']}")
    yml_path  = '_results/cases_chirp_n{}/{}/{}_re{:04.1f}_mean00_A01{:s}.yaml'.format(cs['n'], cs['airfoil_name'], cs['airfoil_name'], cs['re'], cs['suffix'])
    json_path = yml_path.replace('.yaml','.json')
    dvr_path  = yml_path.replace('.yaml', '_UAA.dvr')
    cfd_outb  = yml_path.replace('.yaml', '_CFD.outb')

    # --- JSON Info
    info, dfm    = load_json_chirp(json_path, verbose = False, plot = False)
    info.update(cs)
    info['scale0'] = False
    info['log'] = True


#     # ---CFD
#     try:
    dfc = FASTOutputFile(cfd_outb).toDataFrame()
    _, trc, stc, chc, dwc = split_chirp(info, dfm, dfc, plot=False)
    fig=None
    fig, out = postpro_chirp_tf(chc, dwc, info, st=stc, plot=True, fig=fig, label='CFD')
#     if len(out['dw_h'][0]['mag']==0):
#         print('[WARN] CFD EMPTY')
#     except:
#         print('>>> CFD FAIL', cfd_outb)
#         pass
    print('Transient end Cl=', trc['cl'][-2], -info['Cl_alpha']*np.radians(info['alpha0']))

#     plotDatAgg(ax, datUA[2], label='UA2', color=fColrs(2), ls='--')
#     plotDatAgg(ax, datUA[3], label='UA3', color=fColrs(3), ls='--')
#     plotDatAgg(ax, datUA[5], label='UA5', color=fColrs(4), ls='--', lw=2)
#     plotDatAgg(ax, datULS, label='ULS', color=(0.3,0.3,0.3), lw=2.5, ls=':')

    uaa_outb  = yml_path.replace('.yaml', '_UA2_OF.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA2 OF', c=fColrs(2))

    uaa_outb  = yml_path.replace('.yaml', '_UA3_OF.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA3 OF', c=fColrs(3))

    uaa_outb  = yml_path.replace('.yaml', '_UA5_OF.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA5 OF', c=fColrs(4))

#     uaa_outb  = yml_path.replace('.yaml', '_UA4_OF.outb'); 
#     dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
#     _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
#     fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA4 OF', c=python_colors(0), scale0=scale0,log=log, ls='', marker='.')

#     uaa_outb  = yml_path.replace('.yaml', '_UA5_Wg.outb'); 
#     dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
#     _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
#     fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA5 Wg', c=python_colors(1))
# 
#     uaa_outb  = yml_path.replace('.yaml', '_UA5_Ks.outb'); 
#     dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
#     _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
#     fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA5 Ks', c=python_colors(2))

#     uaa_outb  = yml_path.replace('.yaml', '_UA3_OF.outb'); 
#     dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
#     _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
#     fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA3 OF', c=python_colors(3))


    # --- Theory
    f_th = out['f']
    pTh = pWag
    out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=info['Cl_alpha'], A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=info['chord'])
    plotmelog(fig.axes[0], out['k'], out_th['mag'], info, label='Theory Wag', c='k')
    fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory Wag', c='k')

#     pTh = pOF
#     out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=info['Cl_alpha'], A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=chord)
#     plotmelog(fig.axes[0], out['k'], out_th['mag'], info, label='Theory OF', c='k', ls='--')
#     fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory OF', c='k', ls='--')
# 
#     pTh = pKus
#     out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=info['Cl_alpha'], A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=chord)
#     plotmelog(fig.axes[0], out['k'], out_th['mag'], info, label='Theory Kus', c='k', ls=':')
#     fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory Kus', c='k', ls=':')

    fig.suptitle(base.replace('_',' '))

    fig.axes[0].legend(ncol=3, fontsize=10)



#     postpro_cycles_loops(dw, info)



plt.show()






