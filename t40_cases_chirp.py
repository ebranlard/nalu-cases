import os
import numpy as np
import glob
import json
import yaml

#import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import chirp, butter, filtfilt
from scipy.optimize import curve_fit


from nalulib import pyhyp
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus_rotate import exo_rotate
from nalulib.exodus_quads2hex import exo_zextrude



# --- Main inputs
nSpan = 2
nT_steady       = 10  # TODO 10 then 50
K_TARGET        = 0.5 # TODO 0.6 or 1
F0_FACTOR       = 1   # TODO 2 or 5
N_CYCLES_DWELLS = 1   # TODO 5
NCONV           = 20  # TODO 10 then 250
NCYCLES_CHIRP   = 2   # TODO 2
DT_FACT         = 0.05 # TODO 0.02
PREFIX='SHORT_'
# 
# nSpan = 2
# nT_steady       = 50  # TODO 10 then 50
# K_TARGET        = 0.6 # TODO 0.6 or 1
# F0_FACTOR       = 2   # TODO 2 or 5
# N_CYCLES_DWELLS = 5   # TODO 5
# NCONV           = 250  # TODO 10 then 250
# NCYCLES_CHIRP   = 2   # TODO 2
# DT_FACT         = 0.05 # TODO 0.02
# PREFIX=''


nSpan = 4
nT_steady       = 50  # TODO 10 then 50
K_TARGET        = 0.6 # TODO 0.6 or 1
F0_FACTOR       = 2   # TODO 2 or 5
N_CYCLES_DWELLS = 5   # TODO 5
NCONV           = 250  # TODO 10 then 250
NCYCLES_CHIRP   = 2   # TODO 2
DT_FACT         = 0.05 # TODO 0.02
PREFIX=''

nSpan = 24
nT_steady       = 50  # TODO 10 then 50
K_TARGET        = 0.6 # TODO 0.6 or 1
F0_FACTOR       = 2   # TODO 2 or 5
N_CYCLES_DWELLS = 5   # TODO 5
NCONV           = 250  # TODO 10 then 250
NCYCLES_CHIRP   = 2   # TODO 2
DT_FACT         = 0.05 # TODO 0.02
PREFIX=''


nSpan = 121
nT_steady       = 50  # TODO 10 then 50
K_TARGET        = 0.6 # TODO 0.6 or 1
F0_FACTOR       = 2   # TODO 2 or 5
N_CYCLES_DWELLS = 5   # TODO 5
NCONV           = 250  # TODO 10 then 250
NCYCLES_CHIRP   = 2   # TODO 2
DT_FACT         = 0.05 # TODO 0.02
PREFIX=''







SS_WING_PP = False

#Reynolds =[0.1, 0.5, 0.75, 1, 2, 5, 10]
Alpha_mean = [0] # TODO mesh center at 0.25 is not ready if alpha is not 0
Amplitudes = [1]
#nramp=10
# alpha_mean_deg=0.0
# alpha_amp_deg=1.0
A1, A2 = 0.165, 0.335
B1, B2 = 0.0455, 0.3



mesh_dir    ='meshes'
case_dir    ='cases_chirp_n{}'.format(nSpan)
if SS_WING_PP:
    nalu_template ='_templates/airfoil_name/input_no_output.yaml'
else:
    #nalu_template ='_templates/airfoil_name/input_no_output_no_wing_pp.yaml'
    nalu_template ='_templates/airfoil_name/input_no_wing_pp.yaml'
current_path = os.getcwd()
mem=None
nodes=1
ntasks=None



if 'ebranlard' in current_path: # Unity
    cluster = 'unity'
    batch_template ='_templates/submit-unity.sh'
    ntasks=92 #TODO TODO Unity
    hours={4:16, 121:48}[nSpan]
    nodes={4:1 , 121:1}[nSpan]
elif 'ebranlar' in current_path: # Kestrel
    cluster = 'kestrel'
    #batch_template ='_templates/submit-kestrel.sh'
    batch_template ='_templates/submit-kestrel.sh'
    hours={2:24, 4:48, 22:72, 24:102, 121:202}[nSpan]
    nodes={2:1 , 4:1 , 22:2 , 24:1,   121:1}[nSpan]
else:
    #cluster = 'local'
    #batch_template =None
    cluster = 'bash'
    batch_template ='_templates/submit-bash.sh'
    hours=2


chord=1
DENSITY= 1.2
VISCOSITY = 9.0E-06
SPECIFIC_DISSIPATION_RATE= 114.54981120000002
TURBULENT_KE= 0.0013020495206400003

N = 150
yplus=0.1




# 
db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
db = db.select({'Roughness':'Clean'})
db = db.query('airfoil!="L303"') # No geometry for L303
airfoil_names = db.configs['airfoil'].unique()

# airfoil_names =  list(airfoil_names) + ['du00-w2-212', 'nlf1-0416'] 
# airfoil_names = ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']  +  list(airfoil_names)
airfoil_names = ['S809']
airfoil_names += ['du00-w-212', 'ffa-w3-211', 'nlf1-0416']
# airfoil_names = ['nlf1-0416']
# airfoil_names = ['du00-w-212']

print(f'-------------------------------- SETUP ---------------------------------')
print(f'cluster      : {cluster}')
print(f'hours        : {hours}')
print(f'ntasks       : {ntasks}')
print(f'nodes        : {nodes}')
print(f'airfoil_names: {airfoil_names}')



background_3d = './meshes/background_n{}.exo'.format(nSpan)

if not os.path.exists(background_3d):
    background_3d_n1 = './meshes/background_n1.exo'
    exo_zextrude(background_3d_n1, background_3d, nSpan=nSpan, zSpan=4.0, zoffset=0.0, verbose=True, airfoil2wing=False, ss_wing_pp=False, profiler=False, ss_suffix='_bg')


yml = NALUInputFile(nalu_template)



def determine_chirp_duration(f0, velocity, chord, ncycles=2, nconv=250, verbose=False):
    """
    Calculates chirp duration based on the lowest frequency and convective scales.
    """
    # --- Frequency Constraint: 
    # Ensure we have enough oscillations at the slowest speed to 'see' the physics.
    t_min = ncycles / f0
    # --- Convective Constraint: 
    # Total distance traveled in chord lengths. 
    # For System ID, 400-600 convective time units is a healthy target.
    t_conv = (nconv * chord) / velocity
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
                        nconv=250,
                        chord=1, verbose=True, plot=True):
    # --- Physical Parameters ---
    fs = 1.0 / dt 
    t_conv = chord / U  # Convective time (1 chord)
    s_factor = (2 * U) / chord   
    
    # --- Frequency Calculations ---
    f_break_low = (B1 * U) / (np.pi * chord)
    f0 = f_break_low / f0_factor
    f1 = (k_target * U) / (np.pi * chord)
    f_ref = U / (np.pi * chord)

    
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
    T_chirp = determine_chirp_duration(f0, U, chord, ncycles=NCYCLES_CHIRP, nconv=nconv, verbose=True)
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
    alpha_total = np.concatenate([alpha_transient, alpha_step, alpha_chirp] + dwell_alphas)
    t_total = np.arange(len(alpha_total)) * dt

    # Store comprehensive info for post-processing
    info={
        'dt': dt, 'U': U, 'chord': chord, 's_factor': s_factor,
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

        print(f"Sampling: {fs:.1f} Hz | dt: {dt:.8f} s | U: {U:.2f} m/s")
        print(f"Chirp Range: f0={f0:.3f} Hz to f1={f1:.2f} Hz ({f0/f_ref:.2f} to {f1/f_ref:.2f}, k={k_target})")

    if plot:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(t_total, np.degrees(alpha_total), 'b', lw=1.2)
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

    return t_total, alpha_total, info


def create_case(alpha_mean, amplitude, nT_steady, re, mesh_file_2d, background_3d, nalu_template, sim_dir, basename, nSpan=4, density=1.2, viscosity=9.0e-6, turbulent_ke=TURBULENT_KE, specific_dissipation_rate=SPECIFIC_DISSIPATION_RATE, chord=1, batch_template=None, nramp=5):

    if isinstance(nalu_template, str):
        yml_template = NALUInputFile(nalu_template)
    else:
        yml_template = nalu_template

    sim_dir = os.path.join(case_dir, airfoil_name)
    local_mesh_dir = os.path.join(sim_dir, 'meshes')
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    if not os.path.exists(local_mesh_dir):
        os.makedirs(local_mesh_dir)

    # --- Scales
    U = float(re*1e6 *viscosity /(density * chord ))
    dt = float(np.around(DT_FACT * chord / U, 8))
    T = chord/U*nT_steady


    basename_ReMean = basename+'_'+'re{:04.1f}_mean{:02d}'.format(re, int(alpha_mean))
    basename = PREFIX+basename_ReMean+'_'+'A{:02d}'.format(int(amplitude))
    yaml_file = os.path.join(sim_dir, basename+'.yaml')

    # --- Creating meshes
    rotated_mesh_2d  = os.path.join(local_mesh_dir, basename_ReMean+'_n1.exo')
    extruded_mesh_2d = os.path.join(local_mesh_dir, basename_ReMean+'_n{}.exo'.format(nSpan))
    if not os.path.exists(extruded_mesh_2d):
        if not os.path.exists(rotated_mesh_2d):
            print('Rotating mesh: ', rotated_mesh_2d, alpha_mean)
            exo_rotate(mesh_file_2d, rotated_mesh_2d, angle=alpha_mean, center=(0.5,0), angle_center=(0.5,0), 
                          translation_after=(-0.25,0), # TODO only works for alpha_mean=0 for now, otherwise, we need some kind of cos and sine of 0.25
                          inlet_start=None, inlet_span=None, outlet_start=None, keep_io_side_set=False, 
                          inlet_name='inlet', outlet_name='outlet',
                          verbose=False, profiler=False, debug=False)
        print('Extrudingmesh: ', extruded_mesh_2d, nSpan)
        exo_zextrude(rotated_mesh_2d, extruded_mesh_2d, nSpan=nSpan, zSpan=4.0, zoffset=0.0, verbose=False, airfoil2wing=True, ss_wing_pp=SS_WING_PP, profiler=False, ss_suffix=None)
        try:
            os.remove(rotated_mesh_2d)
        except:
            print('[WARN] Cant delete: ', rotated_mesh_2d)


    # --- Change yaml file
    yml = yml_template.copy()
    # Shortcuts 
    ti = yml.data['Time_Integrators'][0]['StandardTimeIntegrator']
    realms = yml.data['realms']
    bg = realms[0]
    af = realms[1]
    bg['mesh'] = os.path.relpath(background_3d, sim_dir).replace('\\', '/')
    af['mesh'] = os.path.relpath(extruded_mesh_2d, sim_dir).replace('\\', '/')
    if 'restart' in bg:
        bg['restart']['restart_data_base_name'] = 'restart/'+basename+'_bg'
        af['restart']['restart_data_base_name'] = 'restart/'+basename+'_arf'

    if SS_WING_PP:
        pp = af['post_processing'][0]['output_file_name'] = 'forces_'+basename+'_pp.csv'
        pp = af['post_processing'][1]['output_file_name'] = 'forces_'+basename+'.csv'
    else:
        pp = af['post_processing'][0]['output_file_name'] = 'forces_'+basename+'.csv'

    # Flow variables
    yml.velocity = [U, 0, 0]
    yml.density = density
    yml.viscosity = viscosity

    yml.inflow_turbulent_ke               = turbulent_ke
    yml.outflow_turbulent_ke              = turbulent_ke
    yml.IC_turbulent_ke                   = turbulent_ke
    yml.inflow_specific_dissipation_rate  = specific_dissipation_rate
    yml.outflow_specific_dissipation_rate = specific_dissipation_rate
    yml.IC_specific_dissipation_rate      = specific_dissipation_rate




    # --- Motion
    #t_steady = T
    t, theta, info = generate_step_chirp(dt, U, B1, alpha_mean_deg=alpha_mean, alpha_amp_deg=amplitude, 
                            n_chord_transient=nT_steady, n_chord_step=nT_steady, f0_factor=F0_FACTOR, k_target=K_TARGET, 
                            k_dwells=[0.1, 0.3, 0.5],
                            n_cycles_dwells=N_CYCLES_DWELLS, nconv=NCONV,
                            chord=chord, verbose=True, plot=False)
    x=theta*0
    y=theta*0
    yml.set_motion(t, x, y, np.degrees(theta), plot=False, irealm=1)
    T = np.max(t)
    #plt.show()


    # Time
    ti['time_step'] = dt
    ti['termination_step_count'] = int(T/dt)

    # --- Output
    if 'output' in bg:
        #bg['output']['output_data_base_name'] # handled by polar_aseq
        bg['output']['output_frequency'] = int(T/dt)-1
        af['output']['output_frequency'] = int(T/dt)-1
        #bg['output']['output_frequency'] = 1
        #af['output']['output_frequency'] = 1

    if batch_template is not None:
        batch_file = nalu_batch(batch_template, nalu_input_file=yaml_file, jobname='c'+f'_n{nSpan}_'+basename, sim_dir=sim_dir, mail=True, hours=hours, nodes=nodes, mem=mem, ntasks=ntasks)
    else:
        batch_file =None

    print('Saving yaml...')
    yml.save(yaml_file)
    with open(yaml_file.replace('.yaml','.json'), "w") as f:
        json.dump(info, f, indent=2)
    
    return yaml_file, batch_file


# --- Loop through airfoils and create meshes
for ia, airfoil_name in enumerate(airfoil_names):
    print(f'---------------------------- {airfoil_name} ------------------------')
    db_arf = db.select({'airfoil':airfoil_name})

    Reynolds = db_arf.configs['Re'].round(2).unique()
    RE_expected = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 3.0]) # 0.75, 1.0, 1.25, 1.3, 1.4, 1.5]
    RE = []
    for re in Reynolds:
        i=np.argmin(np.abs(re- RE_expected))
        re_ = RE_expected[i]
        RE.append(re_)
    Reynolds=np.array(sorted(list(set(RE))))


    density=DENSITY
    viscosity=VISCOSITY
    specific_dissipation_rate= SPECIFIC_DISSIPATION_RATE
    turbulent_ke=TURBULENT_KE

    # --- HACK ['du00-w-212', 'nlf1-0416', 'ffa'], not in database:
    hack=False
    if len(db_arf)==0:
        hack=True
        if airfoil_name == 'du00-w-212':
            Reynolds=[3]; re=Reynolds[0]
            #mesh_file_2d = './du00-w-212/grids/du00w212_re3M_y03_aoa0_n1.exo'
            mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:04.1f}M_y{yplus}mu.exo')
            density=1.225
            viscosity=1.392416666666667e-05
            #dt_fact=0.55
        elif airfoil_name == 'nlf1-0416':
            Reynolds=[4]; re=Reynolds[0]
            #mesh_file_2d = './nl1-0416/grids/nlf1-0416_re4M_y2_aoa0_n1.exo'
            mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:04.1f}M_y{yplus}mu.exo')
            density=1.225
            viscosity=1.0443125000000002e-05
            specific_dissipation_rate= 460.34999999999997
            turbulent_ke=0.00392448375
            #dt_fact=0.55


        elif airfoil_name == 'ffa-w3-211':
            Reynolds=[10]; re=Reynolds[0]
            #mesh_file_2d = './ffa/grids/ffa_w3_211_near_body_aoa0_n1.exo'
            mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:04.1f}M_y{yplus}mu.exo')
        else:
            raise NotImplementedError(airfoil_name)

    
    #db_arf = db.select({'airfoil':airfoil_name})
    #Reynolds = db_arf.configs['Re'].round(1).unique()
    #print('Reynolds: ', Reynolds, '({})'.format(len(Reynolds)))
    sim_dir = os.path.join(case_dir, airfoil_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)


    nalu_batches = []
    for iRe, re in enumerate(Reynolds):
        mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:04.1f}M_y{yplus}mu.exo')
        for ia, alpha_mean in enumerate(Alpha_mean):
            for iA, amplitude in enumerate(Amplitudes):
                yml, batch = create_case(alpha_mean, amplitude, nT_steady, re, mesh_file_2d, background_3d, yml, sim_dir, basename=airfoil_name, nSpan=nSpan, 
                                         density=density, viscosity=viscosity, turbulent_ke=turbulent_ke, specific_dissipation_rate=specific_dissipation_rate,
                                         batch_template=batch_template)
                nalu_batches.append(batch)
                print('[YML]', yml)
                print('[BAT]', batch)
                if iA==0:
                    break
            if ia==0:
                break
        if iRe==0:
            break

    # --- Write a batch file with all
    #sbatch_file = os.path.join(sim_dir, '_sbatch_all.sh')
    #with open(sbatch_file, 'w', newline="\n") as f:
    #    for b in nalu_batches:
    #        bb = os.path.relpath(b, sim_dir)
    #        prefix='sbatch ' if cluster!='local' else ''
    #        f.write(f'{prefix}{bb}\n')
    #print('SBatch:    ', sbatch_file)

