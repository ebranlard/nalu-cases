import os
import numpy as np
import glob
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_aseq import nalu_aseq
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus_rotate import exo_rotate
from nalulib.exodus_quads2hex import exo_zextrude
from helper_functions import airfoil2configStat

db = DataFrameDatabase('experiments/DB_all_stat.pkl')
db = db.query('airfoil!="L303"') # No geometry for L303
airfoil_names_db = db['airfoil'].unique()

# --- Main inputs
submit=False
aseq = np.arange(-20, 25+3/2, 1)
# aseq = np.arange(-2, 3+3/2, 1)
one_job = False
nT_steady = 200  # leads to approx 10k time steps

airfoil_names = []
airfoil_names += ['S809'] 
# airfoil_names += airfoil_names_db
airfoil_names += ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']

mesh_dir    ='meshes'
case_dir    ='cases_polar'
nalu_template ='_templates/airfoil_name/input_2d_rans.yaml'
current_path = os.getcwd()
mem=None
nodes=1
ntasks=None
if 'ebranlard' in current_path: # Unity
    cluster = 'unity'
    batch_template ='_templates/submit-unity_n1.sh'
    hours=8*len(aseq) if one_job else 8
    ntasks=16
elif 'ebranlar' in current_path: # Kestrel
    cluster = 'kestrel'
    batch_template ='_templates/submit-kestrel_n1.sh'
    hours=0.5
else:
    #cluster = 'local'
    #batch_template =None
    cluster = 'bash'
    batch_template ='_templates/submit-bash.sh'
    hours=2

# TODO TI
N = 150
yplus=0.1

print(f'{f"SETUP":-^70}')
print(f'cluster      : {cluster}')
print(f'hours        : {hours}')
print(f'ntasks       : {ntasks}')
print(f'airfoil_names: {airfoil_names}')


yml_template = NALUInputFile(nalu_template)

# --- Loop through airfoils and create meshes
for ia, airfoil_name in enumerate(airfoil_names):
    print('\n----------------------------------------------------------------------')
    print(f'{airfoil_name:-^70}')
    print('----------------------------------------------------------------------')
    config, db_arf = airfoil2configStat(airfoil_name, db)
    Reynolds = config['Reynolds']
    print('Reynolds: ', Reynolds, '({})'.format(len(Reynolds)))

    for iRe, re in enumerate(Reynolds):
        print(f'{f"Re={re:.2f}":-^70}')

        # --- Main paths and job names
        mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:05.2f}M_y{yplus}mu.exo')
        if not os.path.exists(mesh_file_2d):
            raise Exception('[WARN] Mesh not found: ', mesh_file_2d)

        jobname = airfoil_name + '_re{:05.2f}M'.format(re)
        sim_dir = os.path.join(case_dir, jobname)
        print('sim_dir:   ', sim_dir)
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)
        # --- Scales
        U = float(re*1e6 *config['viscosity'] /(config['density'] * config['chord'] ))
        dt = float(np.around(config['dt_fact'] * config['chord'] / U, 8))
        T = config['chord']/U*nT_steady

        # --- Creating meshes
	# Nothing to do
        # --- Create a input file with proper mesh and flow parameters
        default_yaml_file = os.path.join(sim_dir, 'input.yaml')

        yml = yml_template.copy()
        # Shortcuts 
        ti = yml.data['Time_Integrators'][0]['StandardTimeIntegrator']
        realms = yml.data['realms']
        #bg['mesh'] = os.path.relpath(background_3d, sim_dir).replace('\\', '/')
        #bg = realms[0]
        af = realms[-1]
        af['mesh'] = os.path.relpath(mesh_file_2d, sim_dir).replace('\\', '/')


        # --- Flow variables
        yml.velocity = [U, 0, 0]
        yml.density = config['density']
        yml.viscosity = config['viscosity']

        yml.inflow_turbulent_ke               = config['turbulent_ke']
        yml.outflow_turbulent_ke              = config['turbulent_ke']
        yml.IC_turbulent_ke                   = config['turbulent_ke']
        yml.inflow_specific_dissipation_rate  = config['specific_dissipation_rate']
        yml.outflow_specific_dissipation_rate = config['specific_dissipation_rate']
        yml.IC_specific_dissipation_rate      = config['specific_dissipation_rate']

        # --- Time
        ti['time_step'] = dt
        ti['termination_step_count'] = int(T/dt)
        print('>>>>>>>>>>>', int(T/dt))

        # --- Output and restart
        yml.remove_output()
        yml.remove_restart()

        yml.save(default_yaml_file)
        print('Main yaml: ', default_yaml_file)

        # --- Use nalu_aseq to generate all meshes and input files
        nalu_files, nalu_batches = nalu_aseq(input_file=default_yaml_file,
                  aseq=aseq, 
                  batch_file=batch_template,
                  jobname=f'p_n1_'+jobname+'_',
                  submit=submit,
                  one_job=one_job,
                  raiseError=False, # <<<<<<<<<<<<
                  hours=hours, nodes=nodes, ntasks=ntasks, mem=mem, cluster=cluster)
                  #sim_dir = sim_dir,
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
