import numpy as np
import os
# import pandas as pd
# import json
import matplotlib.pyplot as plt
# from scipy import signal
from scipy.optimize import curve_fit

from welib.weio.fast_output_file import FASTOutputFile

from helper_functions import generate_step_chirp, load_json_chirp, plot_chirp_full_time
from helper_functions import split_chirp
from helper_functions import k2f, f2k

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
        print('[FAIL] fit')
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

# --------------------------------------------------------------------------------}
# --- MAIN SCRIPTS
# --------------------------------------------------------------------------------{
# --- Main inputs
out_dir = '_results/_data_paper/splits/'

cases=[]
# cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':''    }]
cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HR' }]
# cases+=[{'airfoil_name':'du00-w-212' , 'n':4  , 're':3   , 'suffix':''    }]
cases+=[{'airfoil_name':'du00-w-212' , 'n':22 , 're':3   , 'suffix':''    }]
# cases+=[{'airfoil_name':'nlf1-0416'  , 'n':24 , 're':4   , 'suffix':''    }]
# cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':''    }]
cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':'_HR' }]

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

    # All, Transients, Step, Chirp, Dwells
    al, tr, st, ch, dw = split_chirp(info, dfc, dff, plot=False)

    postpro_step(st['t'][5:], st['cl'][5:], info, plot=True) # TODO TODO

    al, tr, st, ch, dw = split_chirp(info, dfc, dfa, plot=False)
    postpro_step(st['t'], st['cl'], info, plot=True) # TODO TODO


plt.show()






