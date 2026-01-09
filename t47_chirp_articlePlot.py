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
from welib.tools.figure import setFigureTitle

from helper_functions import postpro_chirp_tf, load_json_chirp, split_chirp, plotmelog, load_ULS
from ua import get_analytical_tf

setFigurePath('../nalu_torque_2026/figs/')
setFigureFont(12)
setFigureTitle(1)
export=False
export=True

scale0 = True
log    = True

IPlot = []
IPlot += [0] # Comparison of UA


# --------------------------------------------------------------------------------}
# --- MAIN SCRIPTS
# --------------------------------------------------------------------------------{

pWag = [0.165,  0.045, 0.335,  0.300 ] # Wagner / Jones   Pitch change
pKus = [0.500,  0.13 , 0.500,  1.000 ] # Kussner          Transverse Gust
pOF  = [0.3  ,  0.14 , 0.7  ,   0.53 ] # OpenFAST


def aggregate(d0, d1):
#     s_step0, phi_step0, s, phi = d0
    if d1 is None:
        d1 = {}
        d1 = d0
        d1['all']=[d1]
        d1['H_rel'] = d1['H_rel'].reshape((1, -1))
        d1['phi'] = d1['phi'].reshape((1, -1))


    else:
        if len(d0['k'])>len(d1['k']):
               raise Exception('Start with HR')
         # Important interpolate on k
        Hrl2 =  np.interp(d1['k'], d0['k'], d0['H_rel'])
        H2   =  np.interp(d1['k'], d0['k'], d0['H'])
        phi2 =  np.interp(d1['k'], d0['k'], d0['phi'  ] )

        for ii, dwi0 in enumerate(d0['dw_h']):
            Found=False
            for jj, dwi1 in enumerate(d1['dw_h']):
                if dwi1['k'] == dwi0['k']:
                    Found=True
                    break
            if Found is False:
                raise Exception('Dwell not found')
            dwi1['k_vec'] = np.concatenate((dwi1['k_vec'], dwi0['k_vec']))
            dwi1['H_rel'] = np.concatenate((dwi1['H_rel'], dwi0['H_rel']))
            dwi1['phi']   = np.concatenate((dwi1['phi'], dwi0['phi']))



        d1['H_rel'] = np.vstack( (d1['H_rel'], Hrl2))
        d1['H']     = np.vstack( (d1['H']    , H2))
        d1['phi']   = np.vstack( (d1['phi'], phi2))
        d1['all'].append(d0)

#         print(d1['phi'].shape)
    return d1

def plotDatAgg(axes, dat, label=None, color='k', lw=1.5, ls='-', alpha=0.3, dwells=True, bg=True):
    if dat is None:
        print(f'Not plotting {label} at data is None')
        return
    k        = dat['k']
    phimin   = dat['phi'].min(axis  = 0)
    phimax   = dat['phi'].max(axis  = 0)
    phimean  = dat['phi'].mean(axis = 0)
    Hrlmin   = 20*np.log10(dat['H_rel'].min(axis  = 0))
    Hrlmax   = 20*np.log10(dat['H_rel'].max(axis  = 0))
    Hrlmean  = 20*np.log10(dat['H_rel'].mean(axis = 0))
    if bg:
        axes[0].fill_between(k, Hrlmin, Hrlmax, alpha=alpha, color=color)
        axes[1].fill_between(k, phimin, phimax, alpha=alpha, color=color)
#     for i in range(dat['H_rel'].shape[0]):
#         axes[0].plot(        k, 20*np.log10(dat['H_rel'][i]), c=color, label=label, lw=lw, ls=ls)
#     for i in range(dat['phi'].shape[0]):
#         axes[1].plot(        k, dat['phi'][i], c=color, label=label, lw=lw, ls=ls)

    axes[0].plot(        k, Hrlmean, c=color, label=label, lw=lw, ls=ls)
    axes[1].plot(        k, phimean, c=color, label=label, lw=lw, ls=ls)

    if dwells:
        for ii, dwi0 in enumerate(dat['dw_h']):
            axes[0].plot( np.mean(dwi0['k_vec']), np.mean(20*np.log10(dwi0['H_rel'])), c=color, lw=lw, ls='', marker='o', ms=4)
            axes[1].plot( np.mean(dwi0['k_vec']), np.mean(dwi0['phi']), c=color, lw=lw, ls='', marker='o', ms=4)





# --------------------------------------------------------------------------------}
# --- Plot 0 
# --------------------------------------------------------------------------------{

# --- Main inputs
cases=[]
# cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HR' , 'Cl_alpha':6.50842, 'alpha0':  -1.00410  }]
# cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HR' , 'Cl_alpha':6.50842, 'alpha0':  -1.00410  }]
# cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':'_HR',  'Cl_alpha': 6.76063, 'alpha0':-2.78090}] # <<< CFD Not ready

cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':''    , 'Cl_alpha':6.50842, 'alpha0':  -1.00410 }]
cases+=[{'airfoil_name':'du00-w-212' , 'n':22 , 're':3   , 'suffix':'' , 'Cl_alpha':6.71256,  'alpha0':-2.35240}]
cases+=[{'airfoil_name':'nlf1-0416'  , 'n':24 , 're':4   , 'suffix':'' , 'Cl_alpha':6.85040,  'alpha0':-3.93070   }] # <<<< Problem in polar
cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':''    ,  'Cl_alpha': 6.76063, 'alpha0':-2.78090}]


fig=None
doPlot=False

datCFD=None
datULS=None
datUA={2:None, 3:None, 4:None, 5:None, '5OF':None, '5Wg':None, '5Ks':None}

# --- ULS
json_path = '_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01.json'
uls_outb  = '_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_ULS_OmegaM.csv'
uaa_outb  = '_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_UA0_OF.outb'
dfc       = FASTOutputFile(uaa_outb).toDataFrame()
dfl = load_ULS(uls_outb, dfc)
info, dfm = load_json_chirp(json_path, verbose = False, plot = False)
info.update({'Cl_alpha':6.50842, 'alpha0':  -1.00410, 'scale0':scale0, 'log':log})
_, trl, stl, chl, dwl = split_chirp(info, dfm, dfl, plot=False)
fig, out = postpro_chirp_tf(chl, dwl, info, st=stl, plot=doPlot, fig=fig, label='ULS', c=(0.3,0.3,0.3), lw=2.5, ls=':')
datULS = aggregate(out, datULS)


for cs in cases:
    base = cs['airfoil_name'] + '_re{:05.2f}M'.format(cs['re']) + cs['suffix']
    print(f"------------------------- {base}------------- {cs['suffix']}")
    yml_path  = '_results/cases_chirp_n{}/{}/{}_re{:04.1f}_mean00_A01{:s}.yaml'.format(cs['n'], cs['airfoil_name'], cs['airfoil_name'], cs['re'], cs['suffix'])
    json_path = yml_path.replace('.yaml','.json')
    dvr_path  = yml_path.replace('.yaml', '_UAA.dvr')
    cfd_outb  = yml_path.replace('.yaml', '_CFD.outb')

    # --- JSON Info
    info, dfm    = load_json_chirp(json_path, verbose = False, plot = False)
    info.update(cs)
    info['scale0'] = scale0
    info['log']    = log
#     # ---CFD
#     try:
    if True:
        dfc = FASTOutputFile(cfd_outb).toDataFrame()
        _, trc, stc, chc, dwc = split_chirp(info, dfm, dfc, plot=False)
        if len(dwc[-1]['cl'])==0:
            print('[WARN] CFD EMPTY')
        else:
            fig, out = postpro_chirp_tf(chc, dwc, info, st=stc, plot=doPlot, fig=fig, label='CFD')
            print('Transient end Cl=', trc['cl'][-2], -info['Cl_alpha']*np.radians(info['alpha0']))
            datCFD  = aggregate(out, datCFD)
#     except:
#         print('>>> CFD FAIL')
#         pass


    uaa_outb  = yml_path.replace('.yaml', '_UA2_OF.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=doPlot, fig=fig, label='UA2 OF', c=fColrs(2))
    datUA[2] = aggregate(out, datUA[2])

    uaa_outb  = yml_path.replace('.yaml', '_UA3_OF.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=doPlot, fig=fig, label='UA3 OF', c=fColrs(3))
    datUA[3] = aggregate(out, datUA[3])

    uaa_outb  = yml_path.replace('.yaml', '_UA5_OF.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=doPlot, fig=fig, label='UA5 OF', c=fColrs(4))
    datUA[5] = aggregate(out, datUA[5])
    datUA['5OF'] = aggregate(out, datUA['5OF'])

    uaa_outb  = yml_path.replace('.yaml', '_UA5_Wg.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=doPlot, fig=fig, label='UA5 Wg', c=python_colors(1))
    datUA['5Wg'] = aggregate(out, datUA['5Wg'])
# 
    uaa_outb  = yml_path.replace('.yaml', '_UA5_Ks.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=doPlot, fig=fig, label='UA5 Ks', c=python_colors(2))
    datUA['5Ks'] = aggregate(out, datUA['5Ks'])


if doPlot:
    # --- Theory
    f_th = out['f']
    pTh = pWag
    out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=info['Cl_alpha'], A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=info['chord'])
    plotmelog(fig.axes[0], out['k'], out_th['mag'], info, ref=out_th['mag'][1], label='Theory Wag', c='k')
    fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory Wag', c='k')

    # pTh = pOF
    # out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=info['Cl_alpha'], A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=info['chord'])
    # plotmelog(fig.axes[0], out['k'], out_th['mag'], info, ref=out_th['mag'][1], label='Theory OF', c='k', ls='--')
    # fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory OF', c='k', ls='--')
    # 
    # pTh = pKus
    # out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=info['Cl_alpha'], A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=info['chord'])
    # plotmelog(fig.axes[0], out['k'], out_th['mag'], info, ref=out_th['mag'][1], label='Theory Kus', c='k', ls=':')
    # fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory Kus', c='k', ls=':')

    fig.suptitle(base.replace('_',' '))
    axes = fig.axes
    axes[0].set_ylim([-5, 2])
    axes[0].set_xlim(0.05, 1.2)
    axes[1].set_xlim(0.05, 1.2)
    axes[1].set_ylim([-65, 30])
    axes[0].legend(ncol=3, fontsize=10)
    axes[0].set_ylabel(r'Gain $20\log |H|/|H(0)|$ [db]')





# --------------------------------------------------------------------------------}
# --- SuperFig 
# --------------------------------------------------------------------------------{
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.4,4.8))
fig.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.11, hspace=0.04, wspace=0.20)
plotDatAgg(axes, datCFD, label='CFD', color=fColrs(1), ls='-')
plotDatAgg(axes, datULS, label='ULS', color=(0.3,0.3,0.3), lw=2.5, ls=':')
plotDatAgg(axes, datUA[2], label='UA2', color=fColrs(2), ls='--')
plotDatAgg(axes, datUA[3], label='UA3', color=fColrs(3), ls='--')
plotDatAgg(axes, datUA[5], label='UA5', color=fColrs(4), ls='--', lw=2)
axes[0].set_xscale('log')
axes[0].set_ylim([-5, 2])
axes[0].set_xlim(0.05, 1.2)
axes[1].set_xlim(0.05, 1.2)
axes[1].set_ylim([-67, 34])
axes[0].set_ylabel(r'Gain $20\log |H|/|H(0)|$ [db]')
axes[1].set_ylabel(r'Phase $\angle H$ [-]')
axes[1].set_xlabel(r'Reduced frequency $k$ [-]')
axes[0].legend(ncol=3, fontsize=10, loc='lower left')
fig._title='TransferFunctionMultiFidelity'


# --------------------------------------------------------------------------------}
# --- SuperFig 2
# --------------------------------------------------------------------------------{
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.4,4.8))
fig.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.11, hspace=0.04, wspace=0.20)
plotDatAgg(axes,datCFD      ,label = 'CFD'        ,color = fColrs(1),ls = '-'        ,  dwells=False, bg=False)
# plotDatAgg(axes, datUA[3],   label='UA3'         , color=fColrs(3),  ls='--',           dwells=False, bg=False)
plotDatAgg(axes,datUA['5OF'],label = 'UA5'     ,color = fColrs(4),ls = '--',lw = 2 , dwells=False, bg=False)
# plotDatAgg(axes,datUA['5Wg'],label = 'UA5 Wagner' ,color = fColrs(4),ls = ':' ,lw = 2 , dwells=False, bg=False)
# plotDatAgg(axes,datUA['5Ks'],label = 'UA5 Kussner',color = fColrs(4),ls = '-.',lw = 2 , dwells=False, bg=False)
axes[0].set_xscale('log')
axes[0].set_ylim([-5, 2])
axes[0].set_xlim(0.05, 1.2)
axes[1].set_xlim(0.05, 1.2)
axes[1].set_ylim([-67, 34])
axes[0].set_ylabel(r'Gain $20\log |H|/|H(0)|$ [db]')
axes[1].set_ylabel(r'Phase $\angle H$ [-]')
axes[1].set_xlabel(r'Reduced frequency $k$ [-]')
fig._title='TransferFunctionTheoryComp'
# --- Theory
f_th = out['f']
pTh = pWag
out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=info['Cl_alpha'], A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=info['chord'])
plotmelog(fig.axes[0], out['k'], out_th['mag'], info, ref=out_th['mag'][1], label='Theory - Wagner', c='k')
fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory - Wagner', c='k')
pTh = pOF
out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=info['Cl_alpha'], A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=info['chord'])
plotmelog(fig.axes[0], out['k'], out_th['mag'], info, ref=out_th['mag'][1], label='Theory - OpenFAST', c='k', ls='--')
fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory - OpenFAST', c='k', ls='--')
# pTh = pKus
# out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=info['Cl_alpha'], A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=info['chord'])
# plotmelog(fig.axes[0], out['k'], out_th['mag'], info, ref=out_th['mag'][1], label='Theory - Kussner', c='k', ls=':')
# fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory - Kussner', c='k', ls=':')
axes[0].set_ylim([-14, 2])

axes[0].legend(ncol=2, fontsize=10, loc='lower left')














if export:
    export2pdf()




plt.show()








