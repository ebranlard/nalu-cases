import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from welib.weio.fast_output_file import FASTOutputFile
from nalulib.nalu_forces import polar_postpro, standardize_polar_df, plot_polars
from nalulib.nalu_forces_combine import nalu_forces_combine
from nalulib.weio.csv_file import CSVFile

from welib.tools.colors import python_colors

from helper_functions import generate_step_chirp, load_json_chirp, plot_chirp_full_time
from helper_functions import split_chirp
from helper_functions import k2f, f2k

from system_dynamics import tf_from_step, tfestimate, tfestimate_stitched
from system_dynamics import cycle_mag_phase_fit, tf_from_cycle
from ua import get_analytical_tf, compute_bl_response

# --------------------------------------------------------------------------------}
# --- HELPER FUNCTIONS 
# --------------------------------------------------------------------------------{
def fit_wagner_step(s, cl_data, alpha_amp_rad, plot=False):
    """
    Fits the Wagner-like response: Cl(s) = Cl_final * (1 - A1*exp(-b1*s) - A2*exp(-b2*s))
    Commonly used to extract indicial response constants for B-L models.
    """
    # Normalize cl by the steady state change to get the deficiency function
    cl_steady = np.mean(cl_data[-100:])
    
    def wagner_model(s, A1, b1, A2, b2):
        return cl_steady * (1 - A1*np.exp(-b1*s) - A2*np.exp(-b2*s))
    
    # Standard Jones constants as initial guess [A1, b1, A2, b2]
    p0 = [0.165, 0.045, 0.335, 0.30]
    try:
        popt, _ = curve_fit(wagner_model, s, cl_data, p0=p0, maxfev=5000)
        return popt
    except:
        return None


def postpro_step(t, cl, info, plot=False):
    # Detrend to remove DC offsets for better FFT results
    #theta_ch = signal.detrend(theta_ch)
    #cl_ch    = signal.detrend(cl_cl)
    t=np.asarray(t)
    cl=np.asarray(cl)
    t = t.copy()-t[0]
    s_factor = info['s_factor']
    s_step   = t*s_factor
    # --- Postpro
    wagner_params = fit_wagner_step(t*s_factor, cl, np.radians(1.0)) # Example amp
    print(wagner_params)
    if plot:
        ## Plot 1: Step Fit
        plt.figure(figsize=(8, 4))
        plt.plot(s_step, cl, 'k', alpha=0.3, label='CFD')
        if wagner_params is not None:
            plt.plot(s_step, cl[-1]*(1-wagner_params[0]*np.exp(-wagner_params[1]*s_step)-wagner_params[2]*np.exp(-wagner_params[3]*s_step)), 'r--', label='Fit')
        plt.title("Step Response (Wagner Fit)")
        plt.xlabel("s [dim. time]")
        plt.legend()


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
            if np.abs(dd['phi_u'])>np.radians(1) and np.abs(dd['phi_u'])<np.radians(170):
                raise Exception('Error input fit phase')

            mags.append(mag)
            phis.append(phi)

        dw_tf.append({'k': k, 'f': f_target, 'mag':mags, 'phi_deg':np.degrees(phis)})
    return dw_tf





def postpro_chirp_tf(ch, dw, info, st=None, plot=False):
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
    
    A1=0.165; A2 = 0.335; B1= 0.0455; B2=0.3

    # --- Chirp
    Cl_alpha = 6.1
    t_ch, cl_ch, th_ch = ch['t'], ch['cl'], ch['th'] # NOTE: th in radians
    al_ch = -th_ch
    th_deg = np.degrees(th_ch)
    print('B1', info['B1'], B1)
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
    # k = omega * c / (2 * U) -> Note: some use c, some use c/2 (semi-chord)
    k_ch = f2k(f_ch, U, chord)

    # --- BL model
    # TODO TODO Use UA Driver and my own DynamicStall module
    cl_bl = compute_bl_response(t_ch, al_ch, U=U, Cl_alpha=Cl_alpha, A1=A1, A2=A2, B1=B1, B2=B2, MACH=0.0, chord=chord)
    f_bl, H_bl = tfestimate(al_ch, cl_bl, fs, nperseg=None, returnCoh=False, n_pad_factor=n_pad_factor)
    mag_bl = np.abs(H_bl)
    phi_bl = np.angle(H_bl, deg=True)
    k_bl = f2k(f_bl, U, chord)

    # --- Theory
    H_th, H_circ, H_nc = get_analytical_tf(f_ch, U=U, Cl_alpha=Cl_alpha, A1=A1, A2=A2, b1=B1, b2=B2, chord=chord)
    mag_th = np.abs(H_th)
    phi_th = np.angle(H_th, deg=True)



    if dw is not None:
        dw_h = postpro_cycles_tf(dw, info, plot=False)
# 

    if plot:
        # Compare with BL TODO use UA
        # fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8))
        # fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        # ax.plot(t_ch, cl_ch, label='CFD')
        # ax.plot(t_ch, cl_bl, label='BL')
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Cl')
        # ax.legend()

        # Plot 2: Bode Plot
        k0 = f2k(info['f0'], U, chord)
        k1 = f2k(info['f1'], U, chord)
        print('>>> k0', k0, k1)
        mask=k_ch<k1*3
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        # Filter k for plot range (0 to k_target)
        ax1.plot(k_ch[mask], 20 * np.log10(mag_ch[mask]/mag_ch[0]), '.-', label = 'CFD')
        ax1.plot(k_ch[mask], 20 * np.log10(mag_th[mask]/mag_th[0]), label = 'Theory')
        ax1.plot(k_ch[mask], 20 * np.log10(mag_bl[mask]/mag_bl[0]), label = 'BL')
        #mask=Cxy>0.8
        #ax1.plot(k_ch[mask], 20 * np.log10(mag_ch[mask]), '.-', label = 'CFD')
        #ax1.plot(k_ch, 20 * np.log10(mag_th), label = 'Theory')
        #ax1.plot(k_bl, 20 * np.log10(mag_bl), label = 'BL')
        ax1.set_ylabel("Gain [dB]")
        ax1.legend()
        ax2.plot(k_ch[mask], phi_ch[mask])
        ax2.plot(k_ch[mask], phi_th[mask])
        ax2.plot(k_ch[mask], phi_bl[mask])
        ax2.set_ylabel("Phase [deg]")
        ax2.set_xlabel("Reduced Frequency k")

        if dw is not None:
            for dwi in dw_h:
                n = len(dwi['mag'])
                ax1.plot([dwi['k']]*n, 20*np.log10(dwi['mag']/mag_ch[0]), 'k.')
                #phi = np.mod( dwi['phi_deg'], 2*np.pi)-180
                phi =  dwi['phi_deg']
                ax2.plot([dwi['k']]*n, phi, 'k.')

#         if st is not None:
#             ax1.semilogx(k_st, 20 * np.log10(mag_st), label='From step')
#             ax2.semilogx(k_st, phi_st)

        ax1.set_xscale('log')
        ax1.set_xlim(0.05, 2)
        ax2.set_xlim(0.05, 2)


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
out_dir = '_results/_data_paper/splits/'

cases=[]
cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':''    }]
# cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HR' }]
cases+=[{'airfoil_name':'du00-w-212' , 'n':4  , 're':3   , 'suffix':''    }]
cases+=[{'airfoil_name':'du00-w-212' , 'n':22 , 're':3   , 'suffix':''    }]
cases+=[{'airfoil_name':'nlf1-0416'  , 'n':24 , 're':4   , 'suffix':''    }]
cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':''    }]
# cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':'_HR' }]

chord = 1
span  = 4

os.makedirs(out_dir, exist_ok=True)

for cs in cases:
# for cs in [cases[0]]:
    base = cs['airfoil_name'] + '_re{:05.2f}M'.format(cs['re']) + cs['suffix']
    print(f"------------------------- {base}------------- {cs['suffix']}")


    yml_path  = '_results/cases_chirp_n{}/{}/{}_re{:04.1f}_mean00_A01{:s}.yaml'.format(cs['n'], cs['airfoil_name'], cs['airfoil_name'], cs['re'], cs['suffix'])
    json_path = yml_path.replace('.yaml','.json')
    dvr_path  = yml_path.replace('.yaml', '_UAA.dvr')
    cfd_outb  = yml_path.replace('.yaml', '_CFD.outb')
    uaa_outb  = yml_path.replace('.yaml', '_UAA.outb')

    # --- JSON Info
    info, dfc    = load_json_chirp(json_path, verbose = False, plot = False)

    # --- Read postprocessed CFD (see t41_chirp_ua)
    dff = FASTOutputFile(cfd_outb).toDataFrame()
    dfa = FASTOutputFile(uaa_outb).toDataFrame()

    # --- Pretty plot of chirp
    # fig = plot_chirp_full_time(info, dfc, dfm=None, dff=dff, label='CFD')
    # ax = fig.axes[1]
    # ax.plot(dfa['Time_[s]'], dfa['Cl_[-]'], label='UAA')
    # ax.legend()

    def getdf(dd):
        I = dd['I']
        I -=I[0]
        del dd['t2']
        #maxlen = max(len(v) for v in dd.values())
        df = pd.DataFrame({k: pd.Series(v[I]) for k, v in dd.items()})
#         except:
#             import pdb; pdb.set_trace()
        return df

    # --- Split signals
    # All, Transients, Step, Chirp, Dwells
    al, tr, st, ch, dw = split_chirp(info, dfc, dff, plot=False)
    FASTOutputFile().writeDataFrame(df=getdf(al), filename = os.path.join(out_dir, 'all_'+base+'_CFD.outb'), tLabel='t')
    FASTOutputFile().writeDataFrame(df=getdf(tr), filename = os.path.join(out_dir, 'trs_'+base+'_CFD.outb'), tLabel='t')
    FASTOutputFile().writeDataFrame(df=getdf(st), filename = os.path.join(out_dir, 'stp_'+base+'_CFD.outb'), tLabel='t')
    FASTOutputFile().writeDataFrame(df=getdf(ch), filename = os.path.join(out_dir, 'chp_'+base+'_CFD.outb'), tLabel='t')

    postpro_step(st['t'], st['cl'], info, plot=True) # TODO TODO


    al, tr, st, ch, dw = split_chirp(info, dfc, dfa, plot=False)
    FASTOutputFile().writeDataFrame(df=getdf(al), filename = os.path.join(out_dir, 'all_'+base+'_UAA.outb'), tLabel='t')
    FASTOutputFile().writeDataFrame(df=getdf(tr), filename = os.path.join(out_dir, 'trs_'+base+'_UAA.outb'), tLabel='t')
    FASTOutputFile().writeDataFrame(df=getdf(st), filename = os.path.join(out_dir, 'stp_'+base+'_UAA.outb'), tLabel='t')
    FASTOutputFile().writeDataFrame(df=getdf(ch), filename = os.path.join(out_dir, 'chp_'+base+'_UAA.outb'), tLabel='t')

#     for dd in dw:
#         k = dd['k']
#         print(dd['k'], dd.keys())
#         keys = ['t', 'cl', 'cd', 'th', 't2', 'I']
#         d2 = {k: dd[k] for k in keys if k in dd}
#         FASTOutputFile().writeDataFrame(df=getdf(d2), filename = os.path.join(out_dir, f'dw_k{k:3.1f}_'+base+'_CFD.outb'), tLabel='t')

#     FASTOutputFile().writeDataFrame(df=pd.DataFrame(dw), filename = os.path.join(out_dir, 'chp_'+base+'_CFD.outb')



# 
#     # --- Postpros
#     postpro_chirp_tf(ch, dw, info, st=st, plot=True) # Compute TF # TODO also from step
#     postpro_cycles_loops(dw, info)



plt.show()






