import os
import numpy as np
import glob

from nalulib import pyhyp
from nalulib.weio.csv_file import CSVFile
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus_rotate import exo_rotate
from nalulib.exodus_quads2hex import exo_zextrude
from helper_functions import generate_step_chirp
from helper_functions import airfoil2configStat
import json



# --- Main inputs
# # nSpan = 24
# # nT_steady       = 50  # TODO 10 then 50
# # K_TARGET        = 0.6 # TODO 0.6 or 1
# # F0_FACTOR       = 2   # TODO 2 or 5
# # N_CYCLES_DWELLS = 5   # TODO 5
# # N_CONV          = 250  # TODO 10 then 250
# # N_CYCLES_CHIRP  = 2   # TODO 2
# # DT_FACT         = 0.05 # TODO 0.02
# # PREFIX=''

# # Higher res
# nSpan = 24
# nT_steady       = 55   
# K_TARGET        = 1.2  
# F0_FACTOR       = 4    
# N_CYCLES_DWELLS = 5    
# N_CONV          = 600  
# N_CYCLES_CHIRP  = 4    
# DT_FACT         = 0.05 # TODO 0.02
# PREFIX=''
# SUFIX='_HR'

nSpan = 4
nT_steady       = 55   
K_TARGET        = 1.2  
F0_FACTOR       = 4    
N_CYCLES_DWELLS = 5    
N_CONV          = 600  
N_CYCLES_CHIRP  = 4    
DT_FACT         = 0.05 # TODO 0.02
PREFIX=''
SUFIX='_HR'


# --- Torque
#db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
#db = db.select({'Roughness':'Clean'})
#db = db.query('airfoil!="L303"') # No geometry for L303
#airfoil_names = db.configs['airfoil'].unique()

# --- NAWEA
case_dir_base = 'cases_polar3d_nawea'
cases = CSVFile('airfoils_data/DB_NAWEA_configs_reduced.csv').toDataFrame()
airfoil_names = cases['airfoil'].unique().tolist()

chord=1
LL = 500
MM = 150
yplus=0.3
dz = 0.03

# airfoil_names =  list(airfoil_names) + ['du00-w2-212', 'nlf1-0416'] 
# airfoil_names = ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']  +  list(airfoil_names)
#airfoil_names = ['S809']
#airfoil_names += ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']
# airfoil_names = ['nlf1-0416']
# airfoil_names = ['du00-w-212']
# airfoil_names = ['fb60']

SS_WING_PP = False

Alpha_mean = [0] # TODO mesh center at 0.25 is not ready if alpha is not 0
Amplitudes = [1]
# alpha_mean_deg=0.0
# alpha_amp_deg=1.0
A1, A2 = 0.165, 0.335
B1, B2 = 0.0455, 0.3 # B1 used to estimate frequency range of chirp


# --- Derived parameters
zSpan = dz*nSpan

mesh_dir    ='_meshes'
case_dir    ='cases_chirp_dz{}_n{}'.format(dz, nSpan)
if SS_WING_PP:
    nalu_template ='_templates/airfoil_name/input_no_output.yaml'
else:
    nalu_template ='_templates/airfoil_name/input_no_wing_pp.yaml'
current_path = os.getcwd()

# --- Sim inputs
mem=None
nodes=1
ntasks=None
if 'ebranlard' in current_path: # Unity
    cluster = 'unity'
    batch_template ='_templates/submit-unity_n1.sh'
    ntasks=92 #TODO TODO Unity
    hours={4:16, 121:48}[nSpan]
    nodes={4:1 , 121:1}[nSpan]
elif 'ebranlar' in current_path: # Kestrel
    cluster = 'kestrel'
    #batch_template ='_templates/submit-kestrel.sh'
    batch_template ='_templates/submit-kestrel.sh'
    hours={2:24, 4:48, 22:144, 24:144, 121:202}[nSpan]
    nodes={2:1 , 4:1 , 22:1 , 24:1,   121:1}[nSpan]
else:
    #cluster = 'local'
    #batch_template =None
    cluster = 'bash'
    batch_template ='_templates/submit-bash.sh'
    hours=2



print(f'-------------------------------- SETUP ---------------------------------')
print(f'cluster      : {cluster}')
print(f'hours        : {hours}')
print(f'ntasks       : {ntasks}')
print(f'nodes        : {nodes}')
print(f'airfoil_names: {airfoil_names}')



background_3d = os.path.join(mesh_dir, 'background_dz{}_n{}.exo'.format(dz, nSpan))
if not os.path.exists(background_3d):
    background_3d_n1 = os.path.join(mesh_dir, 'background_n1.exo')
    exo_zextrude(background_3d_n1, background_3d, nSpan=nSpan, zSpan=zSpan, zoffset=0.0, verbose=True, airfoil2wing=False, ss_wing_pp=False, profiler=False, ss_suffix='_bg')


yml = NALUInputFile(nalu_template)



DENSITY= 1.2
VISCOSITY = 9.0E-06 # mu
SPECIFIC_DISSIPATION_RATE= 114.54981120000002
TURBULENT_KE= 0.0013020495206400003
def create_case(alpha_mean, amplitude, nT_steady, re, mesh_file_2d, background_3d, nalu_template, sim_dir, basename, nSpan=4, 
                density=DENSITY, viscosity=VISCOSITY, turbulent_ke=TURBULENT_KE, specific_dissipation_rate=SPECIFIC_DISSIPATION_RATE, 
                chord=1, batch_template=None):

    if isinstance(nalu_template, str):
        yml_in = NALUInputFile(nalu_template)
    else:
        yml_in = nalu_template

    sim_dir = os.path.join(case_dir, airfoil_name)
    local_mesh_dir = os.path.join(sim_dir, 'meshes')
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    if not os.path.exists(local_mesh_dir):
        os.makedirs(local_mesh_dir)

    # --- Scales
    U = float(re*1e6 *viscosity /(density * chord )) # viscosity is mu
    dt = float(np.around(DT_FACT * chord / U, 8))
    T = chord/U*nT_steady

    basename_ReMean = basename+'_'+'re{:05.2f}_mean{:02d}'.format(re, int(alpha_mean))
    basename = PREFIX+basename_ReMean+'_'+'A{:02d}'.format(int(amplitude))+SUFIX
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
        exo_zextrude(rotated_mesh_2d, extruded_mesh_2d, nSpan=nSpan, zSpan=zSpan, zoffset=0.0, verbose=False, airfoil2wing=True, ss_wing_pp=SS_WING_PP, profiler=False, ss_suffix=None)
        try:
            os.remove(rotated_mesh_2d)
        except:
            print('[WARN] Cant delete: ', rotated_mesh_2d)
    else:
        print('[INFO] Mesh exists')

    # --- Change yaml file
    yml = yml_in.copy()
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
        af['post_processing'][0]['output_file_name'] = 'forces_'+basename+'_pp.csv'
        af['post_processing'][1]['output_file_name'] = 'forces_'+basename+'.csv'
    else:
        af['post_processing'][0]['output_file_name'] = 'forces_'+basename+'.csv'

    # Flow variables
    yml.velocity = [U, 0, 0]
    yml.density = density
    yml.viscosity = viscosity # this is mu

    yml.inflow_turbulent_ke               = turbulent_ke
    yml.outflow_turbulent_ke              = turbulent_ke
    yml.IC_turbulent_ke                   = turbulent_ke
    yml.inflow_specific_dissipation_rate  = specific_dissipation_rate
    yml.outflow_specific_dissipation_rate = specific_dissipation_rate
    yml.IC_specific_dissipation_rate      = specific_dissipation_rate

    # --- Motion
    #t_steady = T
    t, theta_rad, info = generate_step_chirp(dt, U, B1, alpha_mean_deg=alpha_mean, alpha_amp_deg=amplitude, 
                            n_chord_transient=nT_steady, n_chord_step=nT_steady, f0_factor=F0_FACTOR, k_target=K_TARGET, 
                            k_dwells=[0.1, 0.3, 0.5, 1.0],
                            n_cycles_dwells=N_CYCLES_DWELLS, n_conv=N_CONV, n_cycles_chirp=N_CYCLES_CHIRP,
                            chord=chord, flip=True, verbose=True, plot=False)
    x=theta_rad*0
    y=theta_rad*0
    yml.set_motion(t, x, y, np.degrees(theta_rad), plot=False, irealm=1)
    T = np.max(t)
    #plt.show()


    # --- Time
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

    # --- Saving info to a json file to help postprocessing
    print('Saving yaml...')
    info['viscosity'] = viscosity # this is mu
    info['density']   = density
    info['re']        = re

    yml.save(yaml_file)
    with open(yaml_file.replace('.yaml','.json'), "w") as f:
        json.dump(info, f, indent=2)
    
    return yaml_file, batch_file


# --- Loop through airfoils and create meshes
nalu_batches = []
for ia, airfoil_name in enumerate(airfoil_names):
    print(f'---------------------------- {airfoil_name} ------------------------')

    config, _ = airfoil2configStat(airfoil_name, cases)
    Reynolds = config['Reynolds']

    sim_dir = os.path.join(case_dir, airfoil_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    for iRe, re in enumerate(Reynolds):
        mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}__l{LL}_m{MM}_n1_re{re:05.2f}M_y{yplus}mu.exo')
        print('re',re, mesh_file_2d)
        for ia, alpha_mean in enumerate(Alpha_mean):
            for iA, amplitude in enumerate(Amplitudes):
                yml, batch = create_case(alpha_mean, amplitude, nT_steady, re, mesh_file_2d, background_3d, yml, sim_dir, basename=airfoil_name, nSpan=nSpan, 
                                         density=config['density'], viscosity=config['viscosity'], turbulent_ke=config['turbulent_ke'], specific_dissipation_rate=config['specific_dissipation_rate'],
                                         batch_template=batch_template)
                nalu_batches.append(batch)
                print('[YML]', yml)
                print('[BAT]', batch)
                if iA==0: # Amplitudes
                    break
            if ia==0: # Angles of atack
                break
        if iRe==0: # Reynolds
            break

    # --- Write a batch file with all
    #sbatch_file = os.path.join(sim_dir, '_sbatch_all.sh')
    #with open(sbatch_file, 'w', newline="\n") as f:
    #    for b in nalu_batches:
    #        bb = os.path.relpath(b, sim_dir)
    #        prefix='sbatch ' if cluster!='local' else ''
    #        f.write(f'{prefix}{bb}\n')
    #print('SBatch:    ', sbatch_file)

# --- Batch submit all
sbatch_file = os.path.join(case_dir, '_submit_all.sh')
with open(sbatch_file, 'w', newline="\n") as f:
    for b in nalu_batches:
        bb = os.path.relpath(b, case_dir)
        reldir = os.path.dirname(bb)
        name   = os.path.basename(bb)
        command = f'cd {reldir:25s} && sbatch {name:>50s}    && cd ..\n'
        print(command)
        f.write(command)
