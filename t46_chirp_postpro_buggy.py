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

from helper_functions import generate_step_chirp, load_json_chirp, plot_chirp_full_time
from helper_functions import split_chirp
from helper_functions import k2f, f2k

from system_dynamics import tf_from_step, tfestimate, tfestimate_stitched
from system_dynamics import cycle_mag_phase_fit, tf_from_cycle
from ua import get_analytical_tf, compute_bl_response


scale0 = True
log    = True
def plotme(ax, x, y, **kwargs):
    if scale0:
        y = y/y[0]
    if log:
        ax.plot(x, 20*np.log(y), **kwargs)
    else:
        ax.plot(x, y, **kwargs)


# --------------------------------------------------------------------------------}
# --- HELPER FUNCTIONS 
# --------------------------------------------------------------------------------{
def postpro_cycles_tf(dw, info, plot=False, A=1):
    chord, U = info['chord'], info['U']
    # Storage for processed results
    dw_tf = []
    for dwi in dw:
        k = dwi['k']
        print('---------- k ', k)
        f_target = k2f(k, U, chord)
        
        cycle_stats = []
        mags = []
        phis = []
        for i, cyc in enumerate(dwi['cycles']):
            # Analyze Input (Theta)
            t, u, y = cyc['t'], -cyc['th'], cyc['cl'] # NOTE: alpha = -theta
            if len(y)!=len(u):
                print(f'[WARN] Cycle incomplete / incoherent nt={len(t)} nu={len(u)} ny={len(y)}')
                continue
            mag, phi, dd =  tf_from_cycle(t, u, y, f_target, plot=plot, sine=False)

            print('Input: A', dd['A_u'], 'phi',np.degrees(dd['phi_u']))
            if np.abs(dd['A_u']-np.radians(A))>1e-4:
                raise Exception('Error input fit magnitude')
            if np.abs(dd['phi_u'])>np.radians(3) and np.abs(dd['phi_u'])<np.radians(176):
                raise Exception('Error input fit phase', dd['phi_u'], np.radians(1))

            mags.append(mag)
            phis.append(phi)

        dw_tf.append({'k': k, 'f': f_target, 'mag':mags, 'phi_deg':np.degrees(phis)})
    return dw_tf

def postpro_chirp_tf(ch, dw, info, st=None, plot=False, label='CFD', fig=None, ls='-', c=fColrs(1), lw=1.5, marker=None):
    # Detrend to remove DC offsets for better FFT results
    #theta_ch = signal.detrend(theta_ch)
    #cl_ch    = signal.detrend(cl_cl)
    #print(wagner_params)
    chord, U = info['chord'], info['U']
    sgn=1
    if not info['flip']:
        sgn=-1
    if st is not None:
        t_st, cl_st = st['t'], st['cl']
        A = np.radians(info['alpha_amp_deg'])
        f_st, mag_st, phi_st = tf_from_step(t_st, cl_st, A)
        # Reduced frequency
        # k = omega * c / (2 * U) -> Note: some use c, some use c/2 (semi-chord)
        k_st = (2 * np.pi * f_st * chord) / (2*U)
    # --- Chirp
    t_ch, cl_ch, th_ch = ch['t'], ch['cl'], ch['th'] # NOTE: th in radians
    al_ch = -th_ch
    th_deg = np.degrees(th_ch)
    print('>>> Min Max', np.min(th_deg), np.max(th_deg))
    al_ch = al_ch - np.mean(al_ch)
    cl_ch = cl_ch - np.mean(cl_ch)
    dt = (t_ch[-1] - t_ch[0]) / (len(t_ch) - 1)
    fs = 1/dt
    # tfestimate equivalent using CSD and Pwelch
    n_pad_factor=1
    f_ch, H_ch, Cxy = tfestimate(al_ch, cl_ch, fs, nperseg=None, returnCoh=True, n_pad_factor=n_pad_factor)
    #f_ch, H_ch, _, _, _, _ = tfestimate_stitched(al_ch, cl_ch, fs=fs, f_stitch=0.1*info['f1'], returnAll=True)
    mag_ch = np.abs(H_ch)
    phi_ch = np.angle(H_ch, deg=True)
    k_ch = f2k(f_ch, U, chord) # k = omega * c / (2 * U) -> Note: some use c, some use c/2 (semi-chord)
    if dw is not None:
        dw_h = postpro_cycles_tf(dw, info, plot=False)
    out=dict()
    out['f']   = f_ch
    out['k']   = k_ch
    out['H']   = mag_ch
    out['phi'] = phi_ch
    out['dw_h'] = phi_ch
    if plot:
        # Plot 2: Bode Plot
        k0 = f2k(info['f0'], U, chord)
        k1 = f2k(info['f1'], U, chord)
        print('>>> k0', k0, k1)
        mask=k_ch<k1*3
        if fig is None:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        else:
            axes = fig.axes
        ax1 = axes[0]
        ax2 = axes[1]
        # --- Magnitude
        # Filter k for plot range (0 to k_target)
        plotme(ax1, k_ch[mask], mag_ch[mask], ls=ls, lw=lw, c=c, label = label, marker=marker)
        ax2.plot(k_ch[mask], phi_ch[mask]                         , ls=ls, lw=lw, c=c, label = label, marker=marker)
        #mask=Cxy>0.8
        if log:
            ax1.set_ylabel("Gain [dB]")
        else:
            ax1.set_ylabel("Gain")
        # --- Phase
        ax1.legend()
        ax2.set_ylabel("Phase [deg]")
        ax2.set_xlabel(r"Reduced Frequency $k$ [-]")

        if dw is not None:
            for ii, dwi in enumerate(dw_h):
                n = len(dwi['mag'])-1
                plotme(ax1, [dwi['k']]*n, dwi['mag'][1:], marker='.', c=c) 
                #phi = np.mod( dwi['phi_deg'], 2*np.pi)-180
                phi =  dwi['phi_deg']
                ax2.plot([dwi['k']]*n, phi[1:], '.', c=c)
        # Stransfer function from step
#         if st is not None:
#             ax1.semilogx(k_st, 20 * np.log10(mag_st), label='From step')
#             ax2.semilogx(k_st, phi_st)

        if log:
            ax1.set_xscale('log')
            if scale0:
                ax1.set_ylim([-10, 5])
        else:
            ax1.set_ylim([0, 30])
            pass
        ax1.set_xlim(0.05, 1.5)
        ax2.set_xlim(0.05, 1.5)
    return fig, out


def postpro_cycles_loops(dw, info):

    ## Plot 3: Hysteresis (Last cycle only for clarity)
    plt.figure(figsize=(6, 6))

    COLRS = python_colors()
    STY=[':','-.','--','-','-',':']


    for i,d in enumerate(dw):
        for j,c in enumerate(d['cycles']):
            try:
                plt.plot(np.degrees(-c['th']), c['cl'], label=f"k={d['k']} cycle {i+1}", c=COLRS[i], ls=STY[j])
            except:
                pass

    plt.xlabel("Alpha [deg]")
    plt.ylabel("Cl [-]")
    plt.title("Hysteresis Loops (Stabilized)")
    plt.legend()





# --------------------------------------------------------------------------------}
# --- MAIN SCRIPTS
# --------------------------------------------------------------------------------{
# --- Main inputs
cases=[]
# cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':''    , 'Cl_alpha':6.48300}]
# cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HR' , 'Cl_alpha':6.48300}]
# cases+=[{'airfoil_name':'du00-w-212' , 'n':4  , 're':3   , 'suffix':''    }]
# cases+=[{'airfoil_name':'du00-w-212' , 'n':22 , 're':3   , 'suffix':''    }]
# cases+=[{'airfoil_name':'nlf1-0416'  , 'n':24 , 're':4   , 'suffix':''    }]
# cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':''    }]
cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':'_HR' }]

pWag = [0.165,  0.045, 0.335,  0.300 ] # Wagner / Jones   Pitch change
pKus = [0.500,  0.13 , 0.500,  1.000 ] # Kussner          Transverse Gust
pOF  = [0.3  ,  0.14 , 0.7  ,   0.53 ] # OpenFAST

chord = 1
span  = 4



# For S809
pTh = pWag
f_ch = np.linspace(0.1, 20, 100)
Cl_alpha = 6.48300; alpha0 = -1.0; 


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

    fig = None



#     # ---CFD
    try:
        dfc = FASTOutputFile(cfd_outb).toDataFrame()
        _, _, stc, chc, dwc = split_chirp(info, dfm, dfc, plot=False)
        fig, out = postpro_chirp_tf(chc, dwc, info, st=stc, plot=True, fig=fig, label='CFD')
    except:
        print('>>> CFD FAIL')
        pass


    uaa_outb  = yml_path.replace('.yaml', '_UA5_OF.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA5 OF', c=python_colors(0))

#     uaa_outb  = yml_path.replace('.yaml', '_UA4_OF.outb'); 
#     dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
#     _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
#     fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA4 OF', c=python_colors(0), ls='', marker='.')

    uaa_outb  = yml_path.replace('.yaml', '_UA5_Wg.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA5 Wg', c=python_colors(1))

    uaa_outb  = yml_path.replace('.yaml', '_UA5_Ks.outb'); 
    dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
    _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
    fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA5 Ks', c=python_colors(2))

#     uaa_outb  = yml_path.replace('.yaml', '_UA3_OF.outb'); 
#     dfu = FASTOutputFile(uaa_outb).toDataFrame(); 
#     _, _, stu, chu, dwu = split_chirp(info, dfm, dfu, plot=False)
#     fig, out = postpro_chirp_tf(chu, dwu, info, st=stu, plot=True, fig=fig, label='UA3 OF', c=python_colors(3))


    f_th = out['f']
    pTh = pOF
#     pTh = pKus

    pTh = pWag
    out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=Cl_alpha, A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=chord)
    plotme(fig.axes[0], out['k'], out_th['mag'], label='Theory Wag', c='k')
    fig.axes[1].plot(out['k'],             out_th['phi']                  , label='Theory Wag', c='k')

    pTh = pOF
    out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=Cl_alpha, A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=chord)
    plotme(fig.axes[0], out['k'], out_th['mag'], label='Theory OF', c='k', ls='--')
    fig.axes[1].plot(out['k'],    out_th['phi'], label='Theory OF', c='k', ls='--')

    pTh = pKus
    out_th = get_analytical_tf(f_th, U=info['U'], Cl_alpha=Cl_alpha, A1=pTh[0], A2=pTh[2], b1=pTh[1], b2=pTh[3], chord=chord)
    plotme(fig.axes[0], out['k'], out_th['mag'], label='Theory Kus', c='k', ls=':')
    fig.axes[1].plot(out['k'],    out_th['phi'] , label='Theory Kus', c='k', ls=':')



    fig.axes[0].legend(ncol=3, fontsize=10)



#     postpro_cycles_loops(dw, info)



plt.show()






