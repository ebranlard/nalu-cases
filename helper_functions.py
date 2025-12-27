import os
import pandas as pd
import glob
import numpy as np
from scipy import signal
from scipy.signal import chirp, butter, filtfilt
import matplotlib.pyplot as plt
import json

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
    config['viscosity']                = 9.0e-06
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
    s_factor = (2 * U) / chord   
    
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

    t_chirp, theta_chirp, info2 = generate_step_chirp(**info)
    df_chirp = pd.DataFrame(data=np.column_stack([t_chirp, np.degrees(theta_chirp)]), columns=['Time_[s]','angle_[deg]'])

    return info, df_chirp



def plot_chirp_full_time(info, dfc, dfm=None, dff=None):

    main_indices = [
        info['indices_phases'][0],
        info['indices_phases'][0]+info['indices_phases'][1],
        info['indices_phases'][0]+info['indices_phases'][1]+info['indices_phases'][2],
    ]



    fig,axes = plt.subplots(2, 1, sharex=True, figsize=(12.8,4.8))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.07, hspace=0.20, wspace=0.20)
    # --- Plot angle
    axes[0].plot(dfc['Time_[s]'], dfc['angle_[deg]'], '-')
    if dfm is not None:
        axes[0].plot(dfm['Time_[s]'], dfm['angle_[deg]'], '--', label='From yaml')
    axes[0].set_ylabel('Pitch [deg]')
    s_factor = info['s_factor']
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim()[0] * s_factor, axes[0].get_xlim()[1] * s_factor)
    ax2.set_xlabel('Dimensionless Time [s]')

    t_total = dfc['Time_[s]'].values
    dwell_info=info['dwells']


    itr = int((main_indices[0])/2)
    ist = int((main_indices[0]+main_indices[1])/2)
    icp = int((main_indices[1]+main_indices[2])/2)

    # tdict = dict(y=info['alpha_mean_deg']-1.0, ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.6))
    tdict = dict(y=info['alpha_mean_deg']-1.07, ha='center', va='bottom', fontsize=11) #, bbox=dict(facecolor='white', alpha=0.6))
    tvline = dict(color='k', ls='--', lw=1.5)

    axes[0].text(t_total[itr], s=f"Transients", **tdict)
    axes[0].text(t_total[ist], s=f"Step"      , **tdict)
    axes[0].text(t_total[icp], s=f"Chirp"     , **tdict)

    for idx in main_indices:
        axes[0].axvline(t_total[idx-1], **tvline)
    # Dwell Boundaries (Iterate through dwell_info)
    for i,d in enumerate(dwell_info):
        # We only need the end boundary for each dwell since the start 
        # is the end of the previous one
        axes[0].axvline(t_total[d['end_idx']-1], **tvline)
        # Label the k-value on the plot
        t_mid = t_total[d['start_idx']] + (d['duration_s'] / 2)
        axes[0].text(t_mid, s=f"k={d['k']}", **tdict)


    axes[0].set_xlim(np.min(t_total), np.max(t_total))

    # --- Plot force coeff
    if dff is not None:
        axes[1].plot(dff['Time']    , dff['Cy']    , label='')

    axes[1].set_xlabel('Time [s]')
    # axes[0].legend(loc='upper left')




def split_chirp(info, dfc, dff, plot=True):
    t2  = dfc['Time_[s]'].values
    th = np.radians(dfc['angle_[deg]'].values)
    t = dff['Time'].values
    cl = dff['Cy'].values
    dt = t[1] - t[0]
    fs = 1.0 / dt
    
    i_trans, i_step, i_chirp = info['indices_phases']

    all = {'t':t, 'cl': cl, 'th':th, 't2':t2}

    # --- PHASE 0: Transients ---
    idx_start = 0
    idx_end   = i_trans-1
    t_tr      = t[idx_start:idx_end]
    cl_tr     = cl[idx_start:idx_end]
    th_tr     = th[idx_start:idx_end]
    t2_tr     = t2[idx_start:idx_end]
    tr = {'t':t_tr, 'cl': cl_tr, 'th':th_tr, 't2':t2_tr, 'I':[idx_start, idx_end]}

    # --- PHASE 1: STEP RESPONSE ---
    idx_start = i_trans-1
    idx_end   = i_trans + i_step
    t_st      = t[idx_start:idx_end]
    cl_st     = cl[idx_start:idx_end]
    th_st     = th[idx_start:idx_end]
    t2_st     = t2[idx_start:idx_end]
    st = {'t':t_st, 'cl': cl_st, 'th':th_st, 't2':t2_st, 'I':[idx_start, idx_end]}

    # --- PHASE 2: CHIRP TRANSFER FUNCTION ---
    idx_start = i_trans + i_step
    idx_end   = idx_start + i_chirp
    t_ch      = t[idx_start:idx_end]
    th_ch     = th[idx_start:idx_end]
    cl_ch     = cl[idx_start:idx_end]
    t2_ch     = t2[idx_start:idx_end]
    info['ICH'] = [idx_start, idx_end]
    ch = {'t':t_ch, 'cl': cl_ch, 'th':th_ch, 't2':t2_ch, 'I':[idx_start, idx_end]}

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
        t_d, th_d, cl_d, t_ref_d = t[s:e], th[s:e], cl[s:e], t2[s:e]
        t_all = np.concatenate((t_all, t_d))
        # Split into cycles accurately
        cycles = []
        for i, (s_rel, e_rel) in enumerate(cycle_bounds):
            c_s = s + s_rel
            c_e = s + e_rel
            cycles.append({'t':t[c_s:c_e], 'th': th[c_s:c_e], 'cl': cl[c_s:c_e], 't2':t2[c_s:c_e], 'I':[c_s, c_e]})
        d_loc = d.copy()
        d_loc.update({'k': d['k'], 'cycles': cycles, 't':t_d, 'cl':cl_d, 'th':th_d, 't2':t_ref_d, 'I':[s,e]})
        dw.append(d_loc)


    deltas = np.unique(np.around(np.diff(t_all),4))
    if any(deltas>np.min(deltas)*1.5):
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
    return all, tr, st, ch, dw
