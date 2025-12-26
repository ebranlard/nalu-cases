import os
import numpy as np
import glob

from nalulib import pyhyp
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus_rotate import exo_rotate
from nalulib.exodus_quads2hex import exo_zextrude
from helper_functions import generate_step_chirp



# --- Main inputs
nSpan = 24
nT_steady       = 50  # TODO 10 then 50
K_TARGET        = 0.6 # TODO 0.6 or 1
F0_FACTOR       = 2   # TODO 2 or 5
N_CYCLES_DWELLS = 5   # TODO 5
N_CONV          = 250  # TODO 10 then 250
N_CYCLES_CHIRP  = 2   # TODO 2
DT_FACT         = 0.05 # TODO 0.02
PREFIX=''

# Higher res
nSpan = 24
nT_steady       = 60   # TODO 10, 50 60
K_TARGET        = 1.2  # TODO 0.6 or 1
F0_FACTOR       = 4    # TODO 2 or 4, 5
N_CYCLES_DWELLS = 5    # TODO 5
N_CONV          = 600  # TODO 250, 400
N_CYCLES_CHIRP  = 4    # TODO 2
DT_FACT         = 0.05 # TODO 0.02
PREFIX=''
SUFIX='_HR'






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
    hours={2:24, 4:48, 22:144, 24:144, 121:202}[nSpan]
    nodes={2:1 , 4:1 , 22:1 , 24:1,   121:1}[nSpan]
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





def create_case(alpha_mean, amplitude, nT_steady, re, mesh_file_2d, background_3d, nalu_template, sim_dir, basename, nSpan=4, 
                density=DENSITY, viscosity=VISCOSITY, turbulent_ke=TURBULENT_KE, specific_dissipation_rate=SPECIFIC_DISSIPATION_RATE, 
                chord=1, batch_template=None, nramp=5):

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
    T = chord/U*nT_steady


    basename_ReMean = basename+'_'+'re{:04.1f}_mean{:02d}'.format(re, int(alpha_mean))
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
    t, theta_rad, info = generate_step_chirp(dt, U, B1, alpha_mean_deg=alpha_mean, alpha_amp_deg=amplitude, 
                            n_chord_transient=nT_steady, n_chord_step=nT_steady, f0_factor=F0_FACTOR, k_target=K_TARGET, 
                            k_dwells=[0.1, 0.3, 0.5, 1.0],
                            n_cycles_dwells=N_CYCLES_DWELLS, n_conv=N_CONV, n_cycles_chirp=N_CYCLES_CHIRP,
                            chord=chord, verbose=True, plot=False)
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

