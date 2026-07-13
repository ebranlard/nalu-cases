import os
import numpy as np
import glob
from nalulib.weio.csv_file import CSVFile
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_aseq import nalu_aseq
from nalulib.exodus_quads2hex import exo_zextrude
from helper_functions import airfoil2configStat

# --- Torque
# db = DataFrameDatabase('experiments/DB_all_stat.pkl')
# db = db.query('airfoil!="L303"') # No geometry for L303
# airfoil_names = db.configs['airfoil'].unique()
# case_dir_base = 'cases_polar3d'
#LL = 600
#MM = 150
#yplus=0.1

# --- NAWEA
mesh_dir      = '_meshes'
case_dir_base = 'cases_polar3d_nawea'
cases = CSVFile('airfoils_data/DB_NAWEA_configs_reduced.csv').toDataFrame()
airfoil_names = cases['airfoil'].unique().tolist()
LL = 500
MM = 150
yplus=0.3


# --- Main inputs
submit=False
nT_steady=60
SS_WING_PP=False
dz = 0.03


# airfoil_names = []
# airfoil_names += ['S809'] 
# airfoil_names += ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']
# airfoil_names += ['snl-ffa-w3-560fb', 'snl-ffa-w3-480fb', 'snl-ffa-w3-420fb']

sim_dirs=[]
for nSpan in [2]:
    #aseq = np.arange(-5, 25+3/2, 2.5)
    aseq = np.arange(-5, 20+3/2, 5)
    #aseq = np.arange(-20, 25+3/2, 5)
    #aseq = np.arange(-2, 3+3/2, 1)
    one_job = False

    zSpan = dz*nSpan
    case_dir    = case_dir_base + '_dz{}_n{}'.format(dz, nSpan)
    if SS_WING_PP:
        nalu_template ='_templates/airfoil_name/input_pp.yaml'
    else:
        nalu_template ='_templates/airfoil_name/input_no_wing_pp.yaml'
    current_path = os.getcwd()
    mem=None
    nodes=1
    ntasks=None
    if 'ebranlard' in current_path: # Unity
        cluster = 'unity'
        batch_template ='_templates/submit-unity_n1.sh'
        ntasks=92 #TODO TODO Unity
        hours={4:16, 24:30, 121:48}[nSpan]
        nodes={4:1 , 24:1 , 121:1}[nSpan]
    elif 'ebranlar' in current_path: # Kestrel
        cluster = 'kestrel'
        batch_template ='_templates/submit-kestrel.sh'
        hours={2:2 ,4:3, 24:8, 121:48}[nSpan]
        nodes={2:1, 4:1, 24:1, 121:1}[nSpan]
    else:
        #cluster = 'local'
        #batch_template =None
        cluster = 'bash'
        batch_template ='_templates/submit-bash.sh'
        hours=2


    # --- SETUP
    print(f'{f"SETUP":-^70}')
    print(f'cluster      : {cluster}')
    print(f'hours        : {hours}')
    print(f'ntasks       : {ntasks}')
    print(f'airfoil_names: {airfoil_names}')


    background_3d = './_meshes/background_dz{}_n{}.exo'.format(dz, nSpan)

    if not os.path.exists(background_3d):
        background_3d_n1 = './_meshes/background_n1.exo'
        exo_zextrude(background_3d_n1, background_3d, nSpan=nSpan, zSpan=zSpan, zoffset=0.0, verbose=True, airfoil2wing=False, ss_wing_pp=SS_WING_PP, profiler=False, ss_suffix='_bg')


    yml_template = NALUInputFile(nalu_template)

    # --- Loop through airfoils and create meshes
    for ia, airfoil_name in enumerate(airfoil_names):
        print('\n----------------------------------------------------------------------')
        print(f'{airfoil_name:-^70}')
        print('----------------------------------------------------------------------')
        config, _ = airfoil2configStat(airfoil_name, cases)
        Reynolds = config['Reynolds']
        print('Reynolds: ', Reynolds, '({})'.format(len(Reynolds)))

        for iRe, re in enumerate(Reynolds):
            print(f'{f"Re={re:.2f}":-^70}')

            # --- Main paths and job names           
            mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}__l{LL}_m{MM}_n1_re{re:05.2f}M_y{yplus}mu.exo')
            if not os.path.exists(mesh_file_2d):
                raise Exception('[WARN] Mesh not found: ', mesh_file_2d)

            jobname = airfoil_name + '_re{:05.2f}M'.format(re)
            sim_dir = os.path.join(case_dir, jobname)
            sim_dirs.append(sim_dir)
            print('sim_dir:   ', sim_dir)
            if not os.path.exists(sim_dir):
                os.makedirs(sim_dir)
            local_mesh_dir = os.path.join(sim_dir, 'meshes')
            if not os.path.exists(local_mesh_dir):
                os.makedirs(local_mesh_dir)


            # --- Scales
            U = float(re*1e6 *config['viscosity'] /(config['density'] * config['chord'] )) # Viscosity is mu not mu
            dt = float(np.around(config['dt_fact'] * config['chord'] / U, 8))
            T = config['chord']/U*nT_steady

            # --- Creating meshes
            extruded_mesh = os.path.join(local_mesh_dir, 'input_mesh'+'_n{}.exo'.format(nSpan))
            if not os.path.exists(extruded_mesh):
                exo_zextrude(mesh_file_2d, extruded_mesh, nSpan=nSpan, zSpan=zSpan, zoffset=0.0, verbose=False, airfoil2wing=True, ss_wing_pp=SS_WING_PP, profiler=False, ss_suffix=None)

            # --- Create a input file with proper mesh and flow parameters
            default_yaml_file = os.path.join(sim_dir, 'input.yaml')

            yml = yml_template.copy()
            # Shortcuts 
            ti = yml.data['Time_Integrators'][0]['StandardTimeIntegrator']
            realms = yml.data['realms']
            bg = realms[0]
            af = realms[1]

            # --- Mesh
            bg['mesh'] = os.path.relpath(background_3d, sim_dir).replace('\\', '/')
            af['mesh'] = os.path.relpath(extruded_mesh, sim_dir).replace('\\', '/')

            # --- Flow variables
            yml.velocity = [U, 0, 0]
            yml.density = config['density']
            yml.viscosity = config['viscosity'] # This is mu

            yml.inflow_turbulent_ke               = config['turbulent_ke']
            yml.outflow_turbulent_ke              = config['turbulent_ke']
            yml.IC_turbulent_ke                   = config['turbulent_ke']
            yml.inflow_specific_dissipation_rate  = config['specific_dissipation_rate']
            yml.outflow_specific_dissipation_rate = config['specific_dissipation_rate']
            yml.IC_specific_dissipation_rate      = config['specific_dissipation_rate']

            # --- Time
            ti['time_step'] = dt
            ti['termination_step_count'] = int(T/dt)


            # --- Output and restart
            yml.remove_output()
            yml.remove_restart()
            #yml.set_output({'output_frequency': int(T/dt)-1})
            #'output_data_base_name'] # handled by polar_aseq
            #if 'restart' in bg:
            #    bg['restart']['restart_data_base_name'] = 'restart/'+jobname+'_bg'
            #    af['restart']['restart_data_base_name'] = 'restart/'+jobname+'_arf'

            yml.save(default_yaml_file)
            print('Main yaml: ', default_yaml_file)

            # --- Use nalu_aseq to generate all meshes and input files
            nalu_files, nalu_batches = nalu_aseq(input_file=default_yaml_file,
                      aseq=aseq, 
                      batch_file=batch_template,
                      jobname=f'p_n{nSpan}_'+jobname+'_',
                      submit=submit,
                      one_job=one_job,
                      raiseError=False, # <<<<<<<<<<<<
                      hours=hours, nodes=nodes, ntasks=ntasks, mem=mem, cluster=cluster)

            try:
                os.remove(default_yaml_file)
            except:
                pass

            # --- Write a batch file with all
            sbatch_file = os.path.join(sim_dir, '_sbatch_all.sh')
            with open(sbatch_file, 'w', newline="\n") as f:
                for b in nalu_batches:
                    bb = os.path.relpath(b, sim_dir)
                    prefix='sbatch ' if cluster!='local' else ''
                    f.write(f'{prefix}{bb}\n')
            print('SBatch:    ', sbatch_file)

            if iRe==0 and len(Reynolds)>1:
                print('[WARN] STOPPING AT ONE REYNOLDS')
                break


# --- Batch submit all
sbatch_file = os.path.join(case_dir, '_submit_all.sh')
with open(sbatch_file, 'w', newline="\n") as f:
    for sd in sim_dirs:
        reldir = os.path.relpath(sd, case_dir)
        command = f'cd {reldir:25s} && bash _sbatch_all.sh && cd ..\n'
        print(command)
        f.write(command)
