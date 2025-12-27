import os
import numpy as np
import glob
from nalulib import pyhyp
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus_rotate import exo_rotate
from nalulib.exodus_quads2hex import exo_zextrude
import json


# --- Main inputs
nSpan = 4
nSpan = 24
nT_steady=60
nT_oscill=15

DT_FACT         = 0.05 # TODO 0.02
SS_WING_PP = False


alpha_cut=9 # Cutoff underwhich we do now try the loop



mesh_dir    ='meshes'
case_dir    ='cases_pitch_n{}'.format(nSpan)
if SS_WING_PP:
    nalu_template ='_templates/airfoil_name/input_no_output.yaml'
else:
    nalu_template ='_templates/airfoil_name/input_no_output_no_wing_pp_no_restart.yaml'

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
    raise
elif 'ebranlar' in current_path: # Kestrel
    cluster = 'kestrel'
    #batch_template ='_templates/submit-kestrel.sh'
    batch_template ='_templates/submit-kestrel.sh'
    hours={4:48, 24:48, 121:202}[nSpan]
    nodes={4:1 , 24:1,   121:1}[nSpan]
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
# airfoil_names += ['du00-w-212', 'ffa-w3-211', 'nlf1-0416']
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


def create_pitching_case(alpha_mean, amplitude, frequency, re, mesh_file_2d, background_3d, nalu_template, sim_dir, basename, nSpan=4, 
                         density=DENSITY, viscosity=VISCOSITY, turbulent_ke=TURBULENT_KE, specific_dissipation_rate=SPECIFIC_DISSIPATION_RATE, 
                         chord=1, mean_round=None, re_round=None, batch_template=None, nT_steady=20, nT_oscill=10):
    if mean_round is None:
        mean_round = alpha_mean
    if re_round is None:
        re_round = re

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
    U = float(re*1e6 *viscosity /(density * chord ))
    dt = float(np.around(DT_FACT * chord / U, 8))
    T = float(1/frequency)

    T_steady = chord/U*nT_steady


    basename_ReMean = basename+'_'+'re{:04.1f}_mean{:04.1f}'.format(re_round, mean_round)
    basename = basename_ReMean+'_'+'A{:04.1f}_f{:03.1f}'.format(amplitude, frequency)
    yaml_file = os.path.join(sim_dir, basename+'.yaml')

    # --- Creating meshes
    rotated_mesh_2d  = os.path.join(local_mesh_dir, basename_ReMean+'_n1.exo')
    extruded_mesh_2d = os.path.join(local_mesh_dir, basename_ReMean+'_n{}.exo'.format(nSpan))
    if not os.path.exists(extruded_mesh_2d):
        if not os.path.exists(rotated_mesh_2d):
            print('Rotating mesh: ', rotated_mesh_2d, alpha_mean)
            x_QC, y_QC = (0.5-0.25*np.cos(np.radians(alpha_mean))), np.sin(np.radians(alpha_mean))*0.25
            print('Quarter Chord after rot: ', x_QC, y_QC)
            exo_rotate(mesh_file_2d, rotated_mesh_2d, angle=alpha_mean, center=(0.5,0), angle_center=(0.5,0), 
                           #translation_after=(-0.25,0), # TODO only works for alpha_mean=0 for now, otherwise, we need some kind of cos and sine of 0.25
                          translation_after=( -x_QC, -y_QC),
                          inlet_start=None, inlet_span=None, outlet_start=None, keep_io_side_set=False, 
                          inlet_name='inlet', outlet_name='outlet',
                          verbose=False, profiler=False, debug=False)
        print('Extrudingmesh: ', extruded_mesh_2d, nSpan)
        exo_zextrude(rotated_mesh_2d, extruded_mesh_2d, nSpan=nSpan, zSpan=4.0, zoffset=0.0, verbose=False, airfoil2wing=True, ss_wing_pp=SS_WING_PP, profiler=False, ss_suffix=None)
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
        bg['restart']['restart_data_base_name'] = 'restart/'+basename_ReMean+'_bg'
        af['restart']['restart_data_base_name'] = 'restart/'+basename_ReMean+'_arf'
   
    if SS_WING_PP:
        pp = af['post_processing'][0]['output_file_name'] = 'forces_'+basename+'_pp.csv'
        pp = af['post_processing'][1]['output_file_name'] = 'forces_'+basename+'.csv'
    else:
        pp = af['post_processing'][0]['output_file_name'] = 'forces_'+basename+'.csv'

    # --- Flow variables
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
    t_steady = np.max([2*T, 100*dt, T_steady])
    t, x, y, theta = yml.set_sine_motion(A=amplitude, f=frequency, n_periods=nT_oscill, t_steady=t_steady, dt=dt, DOF='pitch', irealm=1)
    T = np.max(t)

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
        batch_file = nalu_batch(batch_template, nalu_input_file=yaml_file, jobname='o'+f'_n{nSpan}_'+basename, sim_dir=sim_dir, mail=True, hours=hours, nodes=nodes, mem=mem, ntasks=ntasks)
    else:
        batch_file =None



    print('Saving yaml...')
    yml.save(yaml_file)
    
    info=dict(alpha_mean=alpha_mean, amplitude=amplitude, frequency=frequency, re=re, mesh_file_2d=mesh_file_2d, basename=basename, nSpan=nSpan, 
                         density=density, viscosity=viscosity, turbulent_ke=turbulent_ke, specific_dissipation_rate=specific_dissipation_rate, 
                         chord=chord, mean_round=mean_round, re_round=re_round, nT_steady=nT_steady, nT_oscill=nT_oscill, SS_WING_PP=SS_WING_PP)
    with open(yaml_file.replace('.yaml','.json'), "w") as f:
        json.dump(info, f, indent=2)
    return yaml_file, batch_file

# --- Loop through airfoils and create meshes
for ia, airfoil_name in enumerate(airfoil_names):
    print(f'---------------------------- {airfoil_name} ------------------------')
    db_arf = db.select({'airfoil':airfoil_name})
    db_arf = db_arf.query("Mean<{}".format(alpha_cut))

    Reynolds = db_arf.configs['Re'].round(2).unique()
    Reynolds = np.sort(Reynolds)

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

    print('Reynolds: ', Reynolds, '({})'.format(len(Reynolds)))
    sim_dir = os.path.join(case_dir, airfoil_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    nalu_batches = []
    for iRe, re in enumerate(Reynolds):
        mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:04.1f}M_y{yplus}mu.exo')
        db_re = db_arf.select({'Re':re})
        print(f'---------------------------- Re={re}  - n={len(db_re.configs)}')
        for idx, config in db_re.configs.iterrows():
            alpha_mean = config['mean_real']
            mean_round = config['Mean']
            amplitude = config['Amplitude']
            freq = config['Frequency']
            re_real = config['Re']
            yml, batch = create_pitching_case(alpha_mean, amplitude, freq, re_real, mesh_file_2d, background_3d, yml, sim_dir, basename=airfoil_name, nSpan=nSpan,
                                   density=density, viscosity=viscosity,  turbulent_ke=turbulent_ke, specific_dissipation_rate=specific_dissipation_rate,
                                   mean_round=mean_round, re_round=re, batch_template=batch_template, nT_steady=nT_steady, nT_oscill=nT_oscill)
            nalu_batches.append(batch)
            print('[YML]', yml)
            print('[BAT]', batch)
#             if idx==1:
#                 break
#         if iRe==0:
#             break

    # --- Write a batch file with all
    sbatch_file = os.path.join(sim_dir, '_sbatch_all.sh')
    with open(sbatch_file, 'w', newline="\n") as f:
        for b in nalu_batches:
            bb = os.path.relpath(b, sim_dir)
            prefix='sbatch ' if cluster!='local' else ''
            f.write(f'{prefix}{bb}\n')
    print('SBatch:    ', sbatch_file)

