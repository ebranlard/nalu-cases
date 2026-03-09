import os
import pandas as pd
import glob
import numpy as np
from scipy import signal
from scipy.signal import chirp, butter, filtfilt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json
from welib.weio.csv_file import CSVFile
from welib.weio.fast_output_file import FASTOutputFile
from welib.tools.pandalib import pd_interp1
from welib.tools.colors import fColrs

from system_dynamics import tf_from_step, tfestimate, tfestimate_stitched
from system_dynamics import cycle_mag_phase_fit, tf_from_cycle
from ua import get_analytical_tf, compute_bl_response


try:
    from welib.tools.colors import python_colors
except:
    pass

def airfoil2configStat(airfoil_name, db, verbose=False):
    # Current baseline...
    config={}
    config['airfoil']                  = airfoil_name
    config['chord']                    = 1
    config['density']                  = 1.2
    config['viscosity']                = 9.0e-06 # This is mu, not nu
    config['specific_dissipation_rate']= 114.54981120000002
    config['turbulent_ke']             = 0.0013020495206400003
    config['dt_fact']                  = 0.02

    # --- KEEP ME
    # irfoil_name = 'du00-w-212'
    #config={'airfoil':airfoil_name, 'Re':3, 'density':1.225, 'viscosity':1.392416666666667e-05}
    #airfoil_name = 'nlf1-0416'
    #config={'airfoil':airfoil_name, 'Re':4, 'density':1.225, 'viscosity':1.0443125000000002e-05}
    #config['specific_dissipation_rate'] = 460.34999999999997
    #config['turbulent_ke']              = 0.00392448375

    db_arf = db.select({'airfoil':airfoil_name})
    config_db = db_arf.common_config
    if verbose:
        print('Config_db', config_db)
    config['Reynolds'] = db_arf['Re'].round(2).sort_values().unique()
    for k,v in config_db.items():
        if isinstance(v, (int, float)):
            if not np.isnan(v) and k not in ['Re']:
                print(f'Setting {k:25s}={v}')
                if isinstance(v, int):
                    config[k] = int(v)
                else:
                    config[k] = float(v)

        else:
            config[k] = v


    return config, db_arf







# --------------------------------------------------------------------------------}
# --- Reduce frequency
# --------------------------------------------------------------------------------{
# k = omega * c / (2 * U) -> Note: we use c, others use c/2
# k = pi * f* c / U
#     omega = 2 pi f
def k2f(k, U, chord):
    return (k * U) / (np.pi * chord) # [Hz]

def f2k(f, U, chord):
    return  (np.pi * f * chord) / U  # [-]



# --------------------------------------------------------------------------------}
# --- Chirp 
# --------------------------------------------------------------------------------{

def determine_chirp_duration(f0, velocity, chord, n_cycles=2, n_conv=250, verbose=False):
    """
    Calculates chirp duration based on the lowest frequency and convective scales.
    """
    # --- Frequency Constraint: 
    # Ensure we have enough oscillations at the slowest speed to 'see' the physics.
    t_min = n_cycles / f0
    # --- Convective Constraint: 
    # Total distance traveled in chord lengths. 
    # For System ID, 400-600 convective time units is a healthy target.
    t_conv = (n_conv * chord) / velocity
    # Take the maximum of the two to be safe
    duration = max(t_min, t_conv)
    if verbose:
        print(f"Chirp duration based on cycles at f0    : {t_min:.2f} s")
        print(f"Chirp duration based on convective units: {t_conv:.2f} s")
    return duration


def generate_step_chirp(dt, U, B1, alpha_mean_deg=2.0, alpha_amp_deg=1.0, 
                        n_chord_transient=50, n_chord_step=60, f0_factor=5.0, k_target=1.0, 
                        k_dwells=[0.1, 0.3, 0.5],
                        n_cycles_dwells=4,
                        n_conv=250,
                        n_cycles_chirp=2,
                        chord=1, verbose=True, plot=True, flip=True, **kwargs):
    if verbose:
        for k,v in kwargs.items():
            print('Key unused: ', k)
    # --- Physical Parameters ---
    fs = 1.0 / dt 
    t_conv = chord / U  # Convective time (1 chord)
    s_factor = (2 * U) / chord    # Using semi chord
    
    # --- Frequency Calculations ---
    f_break_low = (B1 * U) / (np.pi * chord)       # [Hz]
    f0          = f_break_low / f0_factor          # [Hz]
    f1          = (k_target * U) / (np.pi * chord) # [Hz]
    f_ref       =             U  / (np.pi * chord) # [Hz]

    
    # Nyquist Check
    if f1 >= 0.5 * fs:
        print(f"[WARN] f1 ({f1:.2f} Hz) is too high for dt. Capping at Nyquist.")
        f1 = 0.4 * fs # Safety margin
    
    # --- Phase 1: Transient (Alpha = Mean) ---
    T_transient = n_chord_transient * t_conv
    t_transient = np.arange(0, T_transient, dt)
    alpha_transient = np.ones_like(t_transient) * np.radians(alpha_mean_deg)
    
    # --- Phase 2: Step Phase (Alpha Mean + Amplitude) ---
    T_step = n_chord_step * t_conv
    t_step = np.arange(0, T_step, dt)
    a_start = np.radians(alpha_mean_deg)
    a_end = np.radians(alpha_mean_deg + alpha_amp_deg)
    alpha_step = np.ones_like(t_step) * a_end
    n_ramp = 2 
    if len(alpha_step) > n_ramp:
        ramp_values = np.linspace(a_start, a_end, n_ramp + 1)
        alpha_step[:n_ramp] = ramp_values[1:]
    # --- Compute Step Pitch Rate (alpha_dot) ---
    # Centered difference for the bulk, one-sided for the edges
    alpha_dot = np.gradient(alpha_step, dt)
    # The max pitch rate happens during our ramp:
    alpha_dot_max = (a_end - a_start) / (n_ramp * dt)

    # Smooth the first two steps:
    # Step 0: Midway point
    # Step 1: Reach the end point (already set by ones_like)
    # This creates a ramp: [Mean] -> [Mean + 0.5*Amp] -> [Mean + Amp]
    if len(alpha_step) > 2:
        alpha_step[0] = a_start + (a_end - a_start) * 0.5
    
    # --- Chirp using log scale
    # Duration: Enough to resolve f0 (at least n cycles)
    T_chirp = determine_chirp_duration(f0, U, chord, n_cycles=n_cycles_chirp, n_conv=n_conv, verbose=verbose)
    t_chirp = np.arange(0, T_chirp, dt)
    alpha_chirp_raw = np.radians(alpha_mean_deg) + np.radians(alpha_amp_deg) * signal.chirp( t_chirp, f0=f0, f1=f1, t1=t_chirp[-1], method='logarithmic')
    # Find the last peak index
    # We look for where the signal is at its maximum (within a tiny tolerance)
    peaks, _ = signal.find_peaks(alpha_chirp_raw)
    if len(peaks) > 0:
        last_peak_idx = peaks[-1]
        # Trim arrays to the last peak
        t_chirp = t_chirp[:last_peak_idx + 1]
        alpha_chirp = alpha_chirp_raw[:last_peak_idx + 1]
    else:
        alpha_chirp = alpha_chirp_raw # Fallback if no peak found

    # --- Phase 4: Dwells ---
    dwell_alphas = []
    dwell_info = []
    current_t = t_chirp[-1] + dt
    k_dwells = sorted(k_dwells, reverse=True)
    for k in k_dwells:
        f_target = (k * U) / (np.pi * chord)
        t_seg = np.arange(0, n_cycles_dwells / f_target, dt)
        #print('>>>> len', len(t_seg), k, n_cycles_dwells, f_target, dt)
        # Cosine starts at mean + amplitude
        alpha_seg = np.radians(alpha_mean_deg) + np.radians(alpha_amp_deg) * np.cos(2 * np.pi * f_target * t_seg)
        
        current_combined_len = len(alpha_transient) + len(alpha_step) + len(alpha_chirp) + \
                               sum(len(a) for a in dwell_alphas)
                               
        dwell_info.append({
            'k': k, 
            'f_hz': f_target,
            'start_idx': current_combined_len, 
            'end_idx': current_combined_len + len(alpha_seg),
            'duration_s': t_seg[-1],
            'duration_dim': t_seg[-1] * s_factor
        })
        dwell_alphas.append(alpha_seg)
    
    # --- Concatenation ---
    alpha_total_rad = np.concatenate([alpha_transient, alpha_step, alpha_chirp] + dwell_alphas)
    if flip:
        alpha_total_rad *=-1
    t_total = np.arange(len(alpha_total_rad)) * dt

    # Store comprehensive info for post-processing
    info={ # Inputs
        'dt': dt, 'U': U,
        'B1':B1,
        'alpha_mean_deg':alpha_mean_deg,
        'alpha_amp_deg':alpha_amp_deg,
        'n_chord_transient':n_chord_transient,
        'n_chord_step':n_chord_step,
        'n_cycles_dwells':n_cycles_dwells,
        'f0_factor':f0_factor,
        'k_target':k_target,
        'k_dwells':k_dwells,
        'n_cycles_chirp':n_cycles_chirp,
        'n_conv':n_conv,
        'chord': chord,
        'flip': flip,
         # Derived
        's_factor': s_factor,
        'f0': f0, 'f1': f1,
        'indices_phases': [len(alpha_transient), len(alpha_step), len(alpha_chirp)],
        'dwells': dwell_info,
        't_total': t_total[-1]
    	 }

    if verbose:
        header = f"{'Phase':<15} | {'tStart (s)':<10} | {'iStart (-)':<10} | {'sStart (-)':<10} | {'Dur (s)':<8} | {'Dur (-)':<8}"
        print("\n" + header)
        print("-" * len(header))
        # Helper to print rows
        def print_row(name, t_start, i_start, dur_t):
            print(f"{name:<15} | {t_start:<10.3f} | {i_start:<10} | {t_start*s_factor:<10.1f} | {dur_t:<8.2f} | {int(dur_t/dt):<10}")
        # Static Phases
        print_row("Transients", 0.0, 0, t_transient[-1])
        print_row("Step/Hold", t_total[len(alpha_transient)], len(alpha_transient), t_step[-1])
        i_chirp = len(alpha_transient) + len(alpha_step)
        print_row("Chirp", t_total[i_chirp], i_chirp, t_chirp[-1])
        # Dwell Phases
        for d in dwell_info:
            print_row(f"Dwell k={d['k']:.2f}", t_total[d['start_idx']], d['start_idx'], d['duration_s'])
        
        print("-" * len(header))
        print_row(f"Final", t_total[-1], len(t_total), t_total[-1])
        print("-" * len(header))

        print(f"Sampling   : {fs:.1f} Hz | dt: {dt:.8f} s | U: {U:.2f} m/s")
        print(f"Chirp Range: f0={f0:.3f} Hz to f1={f1:.2f} Hz ({f0/f_ref:.2f} to {f1/f_ref:.2f}, k={k_target})")
        print(f'Flip       :', flip)

    if plot:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(t_total, np.degrees(alpha_total_rad), 'b', lw=1.2)
        ax1.set_xlabel('Physical Time [s]')
        ax1.set_ylabel('Alpha [deg]', color='b')
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim()[0] * s_factor, ax1.get_xlim()[1] * s_factor)
        ax2.set_xlabel('Dimensionless Time [s]')
        # Phase boundaries
        main_indices = [
            len(alpha_transient), 
            len(alpha_transient) + len(alpha_step), 
            len(alpha_transient) + len(alpha_step) + len(alpha_chirp)
        ]
        for idx in main_indices:
            ax1.axvline(t_total[idx-1], color='r', ls='--', lw=1.5, alpha=0.7)
        # Dwell Boundaries (Iterate through dwell_info)
        for i,d in enumerate(dwell_info):
            # We only need the end boundary for each dwell since the start 
            # is the end of the previous one
            ax1.axvline(t_total[d['end_idx']-1], color='r', ls='--', lw=1.0)
            # Label the k-value on the plot
            t_mid = t_total[d['start_idx']] + (d['duration_s'] / 2)
            ax1.text(t_mid, alpha_mean_deg-i/5, f"k={d['k']}", ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.6))

        plt.title(f"CFD Input: U={U:.1f}m/s, k_target={k_target}")

    return t_total, alpha_total_rad, info

def load_json_chirp(json_path, verbose=False, plot=False):
    import json
    with open(json_path, 'r') as f:
        info = json.load(f)
    info['B1']                = info.get('B1', 0.0455)
    info['n_chord_transient'] = info.get('n_chord_transient',  50)
    info['n_chord_step']      = info.get('n_chord_step'     ,  50)
    info['n_cycles_dwells']   = info.get('n_cycles_dwells',  5)
    info['f0_factor']         = info.get('f0_factor',  2)
    info['k_target']          = info.get('k_target', 0.6)
    info['k_dwells']          = info.get('k_dwells',  [0.5, 0.3, 0.1])
    info['n_conv']            = info.get('n_conv', 250)
    info['n_cycles_chirp']    = info.get('n_cycles_chirp', 2)
    info['alpha_mean_deg']    = info.get('alpha_mean_deg', 0)
    info['alpha_amp_deg']     = info.get('alpha_amp_deg', 1)
    info['flip']              = info.get('flip', False)
    info['verbose']=verbose
    info['plot']=plot

    if 'S809' in json_path:
        info['viscosity'] = info.get('viscosity',  9e-06)
        info['density']   = info.get('density',  1.2)
        if '0.8' in json_path:
            info['re']        = info.get('re',  0.8)
        elif '0.75' in json_path:
            info['re']        = info.get('re',  0.75)
        else:
            raise Exception()
    elif 'du' in json_path:
        info['viscosity'] = info.get('viscosity',  1.392416666666667e-05)
        info['density']   = info.get('density',  1.225)
        info['re']        = info.get('re',  3)
    elif 'nlf' in json_path:
        info['viscosity'] = info.get('viscosity', 1.0443125000000002e-05)
        info['density']   = info.get('density',  1.225)
        info['re']        = info.get('re',  4)
    elif 'ffa' in json_path:
        info['viscosity'] = info.get('viscosity',  9e-06)
        info['density']   = info.get('density',  1.2)
        info['re']        = info.get('re',  10)
    else:
        raise Exception()






    t_chirp, theta_chirp, info2 = generate_step_chirp(**info)
    df_chirp = pd.DataFrame(data=np.column_stack([t_chirp, np.degrees(theta_chirp)]), columns=['Time_[s]','angle_[deg]'])

    return info, df_chirp



def plot_chirp_full_time(info, dfc, dfm=None, dff=None, label='CFD', col=None, dimLessTime=False, HR=False, figsize=(12.8,4.8)):

    main_indices = [
        info['indices_phases'][0],
        info['indices_phases'][0]+info['indices_phases'][1],
        info['indices_phases'][0]+info['indices_phases'][1]+info['indices_phases'][2],
    ]
    t_total = dfc['Time_[s]'].values


    fig,axes = plt.subplots(2, 1, sharex=True, figsize=figsize)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.07, hspace=0.20, wspace=0.20)
    axes[0].set_xlim(np.min(t_total), np.max(t_total))

    # --- Plot angle
    axes[0].plot(dfc['Time_[s]'], dfc['angle_[deg]'], '-', c=fColrs(1))
    if dfm is not None:
        axes[0].plot(dfm['Time_[s]'], dfm['angle_[deg]'], '--', label='From yaml')
    axes[0].set_ylabel('Pitch [deg]')

    # NOTE: s_factor = (2 * U) / chord   
    s_factor = info['s_factor']
#     ax2 = axes[0].twiny()
#     ax2.set_xlim(axes[0].get_xlim()[0] * s_factor, axes[0].get_xlim()[1] * s_factor)
#     ax2.set_xlabel(r'Dimensionless time, $2Ut/c$ [-]')

    dwell_info=info['dwells']


    itr = int((main_indices[0])/2)
    ist = int((main_indices[0]+main_indices[1])/2)
    icp = int((main_indices[1]+main_indices[2])/2)

    # tdict = dict(y=info['alpha_mean_deg']-1.0, ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.6))
#     ax.text(vTime[t_mask][0], offset + p0*k, z_label, color=colors[i],  fontsize=9, fontweight='bold',
#              bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.01))
    tvline = dict(color='k', ls=':', lw=1.2)

    tdict=[]
    if HR:
        tdict.append( dict(y=info['alpha_mean_deg']-1.30, ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=.02)) ) #, c='') )
        tdict.append( dict(y=info['alpha_mean_deg']+1.10, ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=.02)) ) #, c='')
        axes[0].text(t_total[itr], s=f"Trans.", **tdict[1])
        axes[0].text(t_total[ist], s=f"Step"      , **tdict[1])
        axes[0].text(t_total[icp], s=f"Chirp"     , **tdict[1])
    else:
        tdict.append( dict(y=info['alpha_mean_deg']-0.57, ha='center', va='bottom', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=.02), fontweight='bold') ) #, c='') )
        tdict.append( dict(y=info['alpha_mean_deg']-0.57, ha='center', va='bottom', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=.02), fontweight='bold') ) #, c='')
        axes[0].text(t_total[itr], s=f"Transients", **tdict[0])
        axes[0].text(t_total[ist], s=f"Step"      , **tdict[0])
        axes[0].text(t_total[icp], s=f"Chirp"     , **tdict[0])

    for idx in main_indices:
        axes[0].axvline(t_total[idx-1], **tvline)
        axes[1].axvline(t_total[idx-1], **tvline)
    # Dwell Boundaries (Iterate through dwell_info)
    for i,d in enumerate(dwell_info):
        # We only need the end boundary for each dwell since the start 
        # is the end of the previous one
        axes[0].axvline(t_total[d['end_idx']-1], **tvline)
        axes[1].axvline(t_total[d['end_idx']-1], **tvline)
        # Label the k-value on the plot
        t_mid = (t_total[d['start_idx']] + (d['duration_s'] / 2))
        if dimLessTime:
            t_mid = (t_total[d['start_idx']] + (d['duration_s']*info['s_factor'] / 2))
        print('tmid', t_mid, d)
        if HR:
            ii = np.mod(i+1,2)
            if d['k']==1.0:
                d['k']=1
            axes[0].text(t_mid, s=f"k={d['k']}", **tdict[ii])
        else:
            axes[0].text(t_mid, s=f"k={d['k']}", **tdict[0])


    # --- Plot force coeff
    if dff is not None:
        if 'Cl_[-]' in dff:
            key = 'Cl_[-]'
            axes[1].plot(dff['Time_[s]']    , dff[key]    , label=label, c=col)
        else:
            key = 'Cy'
            axes[1].plot(dff['Time']    , dff[key]    , label=label, c=col)

    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel(key.replace('_',' '))
    # axes[0].legend(loc='upper left')
    axes[0].tick_params(direction='in', top=False, right=True, labelright=False, labeltop=False, which='both')
    axes[1].tick_params(direction='in', top=False, right=True, labelright=False, labeltop=False, which='both')
#     ax2.tick_params(direction='in', top=True, bottom=False, right=False, labelright=False, labeltop=True, which='both')

    return fig




def split_chirp(info, dfc, dff, plot=True):
    t2  = dfc['Time_[s]'].values
    th = np.radians(dfc['angle_[deg]'].values)
    t  = dff['Time_[s]'].values
    cl = dff['Cl_[-]'].values
    cd = dff['Cd_[-]'].values
    dt = t[1] - t[0]
    fs = 1.0 / dt
    
    i_trans, i_step, i_chirp = info['indices_phases']

    al = {'t':t, 'cl': cl, 'cd':cd, 'th':th, 't2':t2, 'I':np.arange(0,len(t))}

    # --- PHASE 0: Transients ---
    idx_start = 0
    idx_end   = i_trans-1
    t_tr      = t [idx_start:idx_end]
    cl_tr     = cl[idx_start:idx_end]
    cd_tr     = cd[idx_start:idx_end]
    th_tr     = th[idx_start:idx_end]
    t2_tr     = t2[idx_start:idx_end]
    tr = {'t':t_tr, 'cd': cd_tr, 'cl': cl_tr, 'cd':cd_tr,'th':th_tr, 't2':t2_tr, 'I':np.arange(idx_start, idx_end)}

    # --- PHASE 1: STEP RESPONSE ---
    idx_start = i_trans-1
    idx_end   = i_trans + i_step
    t_st      = t [idx_start:idx_end]
    cl_st     = cl[idx_start:idx_end]
    cd_st     = cd[idx_start:idx_end]
    th_st     = th[idx_start:idx_end]
    t2_st     = t2[idx_start:idx_end]
    st = {'t':t_st, 'cl': cl_st, 'cd':cd_st, 'th':th_st, 't2':t2_st, 'I':np.arange(idx_start, idx_end)}

    # --- PHASE 2: CHIRP TRANSFER FUNCTION ---
    idx_start = i_trans + i_step
    idx_end   = idx_start + i_chirp
    t_ch      = t[idx_start:idx_end]
    th_ch     = th[idx_start:idx_end]
    cl_ch     = cl[idx_start:idx_end]
    cd_ch     = cd[idx_start:idx_end]
    t2_ch     = t2[idx_start:idx_end]
    info['ICH'] = [idx_start, idx_end]
    ch = {'t':t_ch, 'cl': cl_ch, 'cd':cd_ch, 'th':th_ch, 't2':t2_ch, 'I':np.arange(idx_start, idx_end)}

    t_all = np.concatenate([t_tr, t_st, t_ch])

    # --- PHASE 3: DWELL HYSTERESIS (Cycle Splitting) ---
    dw = []
    n_cycles = info['n_cycles_dwells']
    for d in info['dwells']:
        f_target = d['f_hz'] #  f_target = (k * U) / (np.pi * chord)
        k        = d['k']
        t_seg    = np.arange(0, n_cycles/ f_target, info['dt'])
        indices = []
        T = 1.0 / f_target
        indices = [np.argmin(np.abs(t_seg - i*T)) for i in range(n_cycles + 1)]
        cycle_bounds = [(indices[i], indices[i+1]) for i in range(n_cycles)]
        s, e = d['start_idx'], d['end_idx']
        if len(t_seg)!=e-s:
            raise Exception('Problem in dwell')
        t_d, th_d, cl_d, cd_d, t_ref_d = t[s:e], th[s:e], cl[s:e], cd[s:e], t2[s:e]
        t_all = np.concatenate((t_all, t_d))
        # Split into cycles accurately
        cycles = []
        for i, (s_rel, e_rel) in enumerate(cycle_bounds):
            c_s = s + s_rel
            c_e = s + e_rel
            cycles.append({'t':t[c_s:c_e], 'th': th[c_s:c_e], 'cl': cl[c_s:c_e], 'cd':cd[c_s:c_e], 't2':t2[c_s:c_e], 'I':np.arange(c_s, c_e)})
        d_loc = d.copy()
        d_loc.update({'k': d['k'], 'cycles': cycles, 't':t_d, 'cl':cl_d, 'cd':cd_d, 'th':th_d, 't2':t_ref_d, 'I':np.arange(s,e)})
        dw.append(d_loc)


    deltas = np.unique(np.around(np.diff(t_all),4))
    if any(deltas>np.min(deltas)*1.5):
        import pdb; pdb.set_trace()
        print('>>> deltas', deltas)
        raise Exception('Problem in time coverage, gap found')
    if t_all[-1]!=t[-1]:
        raise Exception('Problem in time coverage, last value')

    # --- PLOTTING ---
    if plot:
        COLRS = python_colors()
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12.8,6.8))
        fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        ax=axes[0]
        ax.plot(t    , cl, 'k-')
        ax.plot(t_tr , cl_tr , label = 'transient', c=COLRS[0])
        ax.plot(t_st , cl_st , label = 'step'     , c=COLRS[1])
        ax.plot(t_ch , cl_ch , label = 'chirp'    , c=COLRS[2])
        for i, d in enumerate(dw):
            ax.plot(d['t'], d['cl'], label=f'Cycle {i+1}', c=COLRS[3+i])
        ax=axes[1]
        STY=['-',':','--','-.','-','--',':']
        for i, d in enumerate(dw):
            for j, c in enumerate(d['cycles']):
               ax.plot(c['t'], c['cl'], STY[j], c=COLRS[3+i])


        ax=axes[2]
        ax.plot(t2,     th, 'k-')
        ax.plot(t2_tr , th_tr , label = 'transient', c=COLRS[0])
        ax.plot(t2_st , th_st , label = 'step'     , c=COLRS[1])
        ax.plot(t2_ch , th_ch , label = 'chirp'    , c=COLRS[2])
        for i, d in enumerate(dw):
            ax.plot(d['t2'], d['th'], label=f'Cycle {i+1}', c=COLRS[3+i])
        ax=axes[3]
        STY=['-',':','--','-.','-','--',':']
        for i, d in enumerate(dw):
            for j, c in enumerate(d['cycles']):
               ax.plot(c['t2'], c['th'], STY[j], c=COLRS[3+i])

        ax.set_xlabel('Time [s]')
#         ax.set_ylabel('')
    return al, tr, st, ch, dw



# --------------------------------------------------------------------------------}
# --- HELPER FUNCTIONS For STEP
# --------------------------------------------------------------------------------{
def wagner_model(s, A1, b1, A2, b2):
    """ Jones approximation for the Wagner function """
    return 1 - A1 * np.exp(-b1 * s) - A2 * np.exp(-b2 * s)

def fit_wagner_step(s_step, phi_step, plot=False):
    """
    Fits the Wagner-like response: phi(s) = 1- A1*exp(-b1*s)-A2*exp(-b2*s))
    Commonly used to extract indicial response constants for B-L models.
    """
    
    # Standard Jones constants as initial guess [A1, b1, A2, b2]
    p0 = [0.165, 0.045, 0.335, 0.30]
    #bounds = ([0, 0, 0, 0], [1.0, 2.0, 1.0, 2.0])
    bounds = ([0, 0, 0, 0.1], [1.0, 0.5, 1.0, 1.0])
    popt, pcov = curve_fit(wagner_model, s_step, phi_step, p0=p0, bounds=bounds)
    A1, b1, A2, b2 = popt
    return popt

def normalize_cl(s, cl, cl_before=None, cl_ss=None, ss_fract=0.01, clip_spike=True, s_min=0.9):
    """ 
    INPUTS:
     - ss_fract: steady state fraction
    """
    if cl_before is None:
        raise Exception('Not recommended')
        cl_before = cl[0]

    # Normalize cl by the steady state change to get the deficiency function
    n     = len(cl)
    if cl_ss is None:
        n_ss  = max(int(n*ss_fract), 2)
        cl_ss = np.mean(cl[-n_ss:])

    phi = (cl-cl_before)/(cl_ss-cl_before)

    s_step   = s.copy()
    phi_step = phi.copy()
    if clip_spike:
        mask = (phi <= 1.05) & (phi >= 0.00) 
        phi_step = phi_step[mask]
        s_step   = s_step[mask]
    if s_min>0:
        mask = s_step>s_min
        phi_step = phi_step[mask]
        s_step   = s_step[mask]
    return s_step, phi_step, phi

def analyse_step_with_tStart(time, Cl, tStep=None, dCldt_lim=0.1, tStepEnd=None, s_factor=1, doFit=True, plot=True, fig=None, label='CFD', c='k', ls='-'):
    """ 

    """
    mask = ~np.isnan(Cl)
    time = time[mask]
    Cl   = Cl[mask]


    time = np.asarray(time)
    Cl   = np.asarray(Cl)
    if tStepEnd is None:
        tStepEnd = time[-1]
        iStepEnd = len(time)-1
    else:
        iStepEnd = np.argmin(np.abs(time-tStepEnd))-1

    if tStep is not None:
        iStepGuess = np.argmin(np.abs(time-tStep))
        nMargin = 10
        if iStepGuess>nMargin:
            dCldt = np.gradient(Cl, time)
            I = np.where(np.abs(dCldt[iStepGuess-nMargin:iStepGuess+nMargin])>dCldt_lim)[0]
            if len(I)==0:
                print('Problem gradient Cl')
                plt.figure()
                plt.plot(time[iStepGuess-nMargin:iStepGuess+nMargin], np.abs(dCldt[iStepGuess-nMargin:iStepGuess+nMargin]))
                plt.show()
            iStep0m = iStepGuess-nMargin + I[0] # 0-, just before the step
            iStep0  = iStep0m+1                 # the step has started
            Cl_before = np.mean(Cl[iStep0m-nMargin:iStep0])
        else:
            iStep0  = iStepGuess
            Cl_before = Cl[0]
    else:
        Cl_before = 0

    tStep = time[iStep0]

    s       = (time-tStep)*s_factor
    t_step  = time[iStep0:iStepEnd]
    Cl_step = Cl  [iStep0:iStepEnd]
    s_step =  s   [iStep0:iStepEnd]

    n_ss  = max(int(len(s_step)*0.001), 2)
    Cl_ss = np.mean(Cl_step[-n_ss:])
    phi = (Cl-Cl_before)/(Cl_ss-Cl_before)

    s_step0, phi_step0, _ = normalize_cl(s_step, Cl_step, cl_before=Cl_before, s_min=0.9, cl_ss=Cl_ss)

    # --- Postpro
    if doFit:
        pwag = fit_wagner_step(s_step0, phi_step0)
        A1, b1, A2, b2 = pwag
        print(f"{label:15s}: A1={A1:6.3f}, b1={b1:6.3f} A2={A2:6.3f}, b2={b2:6.3f}")
    else:
#         print('Fit Skipped')
        pwag = None

    if plot:
        ## Plot 1: Step Fit
        if fig is None:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8))
            fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        else:
            ax = fig.axes[0]

        ax.plot(s      , phi     , c=c, ls=':', alpha=0.2)
        ax.plot(s_step0, phi_step0, c=c, ls=ls, alpha=0.6, label=label)
        if pwag is not None:
            s_pos = s[s>=0]
            phi_wag = wagner_model(s_pos, *pwag)# A1, b1, A2, b2):
            ax.plot(s_pos, phi_wag, '--', c='k')
        ax.set_xlabel(r"Dimensionless time, $s$ [-]")
        ax.legend()
        ax.set_xlim([-1, 30])
        ax.set_ylim([-0.1, 1.1])
    else:
        fig = None


    return pwag, fig, (s_step0, phi_step0, s, phi)

def analyse_step(df, info, fig=None, useCl=True, **kwargs):
    if df is None:
        return [], fig, (None,None, None,None)
    tStep    =  info['indices_phases'][0]*info['dt']
    tStepEnd = (info['indices_phases'][0]+info['indices_phases'][1]) * info['dt']
    time = df['Time_[s]'].values
    if useCl:
        Cl   = df['Cl_[-]'].values
    else:
        Cl   = df['Cn_[-]'].values
    return analyse_step_with_tStart(time, Cl, tStep=tStep, tStepEnd=tStepEnd, s_factor=info['s_factor'], fig=fig, **kwargs)


def load_ULS(ulsfile, dfr):
    if not os.path.exists(ulsfile):
        return None
    dfl = CSVFile(ulsfile).toDataFrame()
    dfl['Time_[s]'] = dfl['Time']
    dfl['Cl_[-]'] = dfl['Cl']
    dfl['Cd_[-]'] = dfl['Cd']
    dfl =  pd_interp1(dfr['Time_[s]'], 'Time_[s]', dfl, extrap='nan')
    return dfl



# --------------------------------------------------------------------------------}
# --- HELPER FUNCTIONS Chirp
# --------------------------------------------------------------------------------{
def plotmelog(ax, x, y, info, ref=None, **kwargs):
    if ref is None:
        ref = y[1]
    if 'label' in kwargs.keys():
        print('Value at 0: ', kwargs['label'], ref, y[1], 20*np.log10(y[1]/ref), 'k=', x[0])
    if info['log']:
        if info['scale0']:
            ax.plot(x, 20*np.log10(y/ref), **kwargs)
        else:
            ax.plot(x, 20*np.log10(y), **kwargs)
    else:
        ax.plot(x, y, **kwargs)
def postpro_cycles_tf(dw, info, plot=False, A=1, verbose=False):
    chord, U = info['chord'], info['U']
    # Storage for processed results
    dw_tf = []
    for dwi in dw:
        k = dwi['k']
        if verbose:
            print('---------- k ', k)
        f_target = k2f(k, U, chord)
        
        cycle_stats = []
        mags = []
        phis = []
        for i, cyc in enumerate(dwi['cycles']):
            # Analyze Input (Theta)
            t, u, y = cyc['t'], -cyc['th'], cyc['cl'] # NOTE: alpha = -theta
            if len(y)!=len(u):
                print(f'[WARN] k={k}, cycle {i} incomplete / incoherent nt={len(t)} nu={len(u)} ny={len(y)}')
                continue
            mag, phi, dd =  tf_from_cycle(t, u, y, f_target, plot=plot, sine=False)

            if verbose:
                print('Input: A', dd['A_u'], 'phi',np.degrees(dd['phi_u']))
            if np.abs(dd['A_u']-np.radians(A))>1e-4:
                print ('Error input fit magnitude')
                pass
                #raise Exception('Error input fit magnitude')
            if np.abs(dd['phi_u'])>np.radians(3) and np.abs(dd['phi_u'])<np.radians(176):
                print ('Error input fit phase')
                pass
                #raise Exception('Error input fit phase', dd['phi_u'], np.radians(1))

            mags.append(mag)
            phis.append(phi)

        dw_tf.append({'k': k, 'f': f_target, 'mag':mags, 'phi_deg':np.degrees(phis)})
    return dw_tf

def postpro_chirp_tf(ch, dw, info, st=None, plot=False, label='CFD', fig=None, ls='-', c=fColrs(1), lw=1.5, marker=None , nperseg=None):
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
    #print('>>> Min Max Angle Chirp:', np.min(th_deg), np.max(th_deg))
    print('Alpha op :',  np.mean(al_ch), 0)
    print('Cl    op :',  np.mean(cl_ch), -info['Cl_alpha']*np.radians(info['alpha0']))
    al_ch = al_ch - np.mean(al_ch)
    cl_ch = cl_ch - np.mean(cl_ch)
    dt = (t_ch[-1] - t_ch[0]) / (len(t_ch) - 1)
    fs = 1/dt
    # tfestimate equivalent using CSD and Pwelch
    n_pad_factor=1
    f_ch, H_ch, Cxy = tfestimate(al_ch, cl_ch, fs, nperseg=nperseg, returnCoh=True, n_pad_factor=n_pad_factor, verbose=True)
    print('>>>>>>>>>>> Merging TF at k1=', (2 * np.pi * info['f1'] * chord) / (2*U)*0.50)
    #f_ch, H_ch, _ = tfestimate_stitched(al_ch, cl_ch, fs=fs, f_stitch=0.1*info['f1'], returnAll=False, method='merge_sig')
    #f_ch, H_ch, _ = tfestimate_stitched(al_ch, cl_ch, fs=fs, f_stitch=0.3*info['f1'], returnAll=False, method='concat')
    mag_ch = np.abs(H_ch)
    phi_ch = np.angle(H_ch, deg=True)
    k_ch = f2k(f_ch, U, chord) # k = omega * c / (2 * U) -> Note: some use c, some use c/2 (semi-chord)
    print(f'Mag   DC: {mag_ch[0]:6.3f} Cla={info["Cl_alpha"]:6.3f}   at: f={f_ch[0]:6.3f} Hz - k ={k_ch[0]:6.3f}')
    if dw is not None:
        dw_h = postpro_cycles_tf(dw, info, plot=False)
    out=dict()
    out['f']   = f_ch
    out['k']   = k_ch
    out['H']   = mag_ch
    out['H_rel'] = mag_ch/mag_ch[1]
    out['H_ref'] = mag_ch[1]
    out['phi'] = phi_ch
    out['k0'] =  f2k(info['f0'], U, chord)
    out['k1'] =  f2k(info['f1'], U, chord)
    if dw is not None:
        for ii, dwi in enumerate(dw_h):
            dwi['k_vec'] = [dwi['k']]*(len(dwi['mag'])-1)
            dwi['H_rel'] = dwi['mag'][1:]/mag_ch[1]
            dwi['phi']   = dwi['phi_deg'][1:]
    out['dw_h'] = dw_h

    if plot:
        # Plot 2: Bode Plot
        mask=k_ch<out['k1']*3
        if fig is None:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.4,4.8))
            fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)

        else:
            axes = fig.axes
        ax1 = axes[0]
        ax2 = axes[1]
        # --- Magnitude
        plotmelog(ax1, k_ch[mask], mag_ch[mask], info, ref=mag_ch[1], ls=ls, lw=lw, c=c, label = label, marker=marker)
        ax2.plot(k_ch[mask], phi_ch[mask]                         , ls=ls, lw=lw, c=c, label = label, marker=marker)
        ax1.set_ylabel("Gain [dB]")
        # --- Phase
        ax1.legend()
        ax2.set_ylabel("Phase [deg]")
        ax2.set_xlabel(r"Reduced Frequency $k$ [-]")

        if dw is not None:
            for ii, dwi in enumerate(dw_h):
                n = len(dwi['mag'])-1
                plotmelog(ax1, [dwi['k']]*n, dwi['mag'][1:], info, ref=mag_ch[1], marker='.', c=c) 
                phi =  dwi['phi_deg']
                ax2.plot([dwi['k']]*n, phi[1:], '.', c=c)
        # Transfer function from step
#         if st is not None:
#             ax1.semilogx(k_st, 20 * np.log10(mag_st), label='From step')
#             ax2.semilogx(k_st, phi_st)
        ax1.set_xscale('log')
        if info['scale0']:
            ax1.set_ylim([-10, 5])
        else:
            pass
        ax1.set_xlim(0.05, 1.5)
        ax2.set_xlim(0.05, 1.5)
    return fig, out




