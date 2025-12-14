import os
import numpy as np
import glob
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_aseq import nalu_aseq
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus_rotate import exo_rotate
from nalulib.exodus_quads2hex import exo_zextrude



# --- Main inputs
submit=False
aseq = np.arange(-20, 25+3/2, 1)
# aseq = np.arange(-2, 3+3/2, 1)
one_job = False


mesh_dir    ='meshes'
case_dir    ='cases_polar'
nalu_template ='_templates/airfoil_name/input_2d_rans_du00-w-212.yaml'
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
    hours=2
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
DT_FACT=0.02


# TODO TI
N = 150
yplus=0.1

# 
db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
db = db.select({'Roughness':'Clean'})
db = db.query('airfoil!="L303"') # No geometry for L303
airfoil_names = db.configs['airfoil'].unique()

# airfoil_names =  list(airfoil_names) + ['du00-w2-212', 'nlf1-0416'] 
airfoil_names = ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']  +  list(airfoil_names)

print(f'-------------------------------- SETUP ---------------------------------')
print(f'cluster      : {cluster}')
print(f'hours        : {hours}')
print(f'ntasks       : {ntasks}')
print(f'airfoil_names: {airfoil_names}')


yml_template = NALUInputFile(nalu_template)

All_Batches = []
All_Inputs  = []

# --- Loop through airfoils and create meshes
for airfoil_name in airfoil_names:
    print(f'---------------------------- {airfoil_name} ------------------------')
    db_arf = db.select({'airfoil':airfoil_name})

    Reynolds = db_arf.configs['Re'].round(1).unique()
    density=DENSITY
    viscosity=VISCOSITY
    dt_fact=DT_FACT
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
            dt_fact=0.55
        elif airfoil_name == 'nlf1-0416':
            Reynolds=[4]; re=Reynolds[0]
            #mesh_file_2d = './nl1-0416/grids/nlf1-0416_re4M_y2_aoa0_n1.exo'
            mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:04.1f}M_y{yplus}mu.exo')
            density=1.225
            viscosity=1.0443125000000002e-05
            specific_dissipation_rate= 460.34999999999997
            turbulent_ke=0.00392448375
            dt_fact=0.55


        elif airfoil_name == 'ffa-w3-211':
            Reynolds=[10]; re=Reynolds[0]
            #mesh_file_2d = './ffa/grids/ffa_w3_211_near_body_aoa0_n1.exo'
            mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:04.1f}M_y{yplus}mu.exo')
        else:
            raise NotImplementedError(airfoil_name)
    if airfoil_name not in ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']:
        break


    Reynolds.sort()
    Reynolds = Reynolds[-1::-1]
    print('Reynolds: ', Reynolds, '({})'.format(len(Reynolds)))

    for re in Reynolds:

        # TODO TODO TODO TO REMEMBER NEAR BODY!
        if hack:
            pass
        else:
            mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:04.1f}M_y{yplus}mu.exo')

        if not os.path.exists(mesh_file_2d):
            print('[WARN] Mesh not found: ', mesh_file_2d)
            continue

        jobname = airfoil_name + '_re{:04.1f}M'.format(re)
        sim_dir = os.path.join(case_dir, airfoil_name + '_re{:04.1f}M'.format(re))
        print('sim_dir:   ', sim_dir)
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)

        #mesh_file_2d_rel = os.path.relpath(mesh_file_2d, sim_dir)

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
        U = float(re*1e6 *viscosity /(density * chord ))
        dt = float(np.around(dt_fact * chord / U, 8))
        yml.velocity = [U, 0, 0]
        yml.density = density
        yml.viscosity = viscosity

        yml.inflow_turbulent_ke               = turbulent_ke
        yml.outflow_turbulent_ke              = turbulent_ke
        yml.IC_turbulent_ke                   = turbulent_ke
        yml.inflow_specific_dissipation_rate  = specific_dissipation_rate
        yml.outflow_specific_dissipation_rate = specific_dissipation_rate
        yml.IC_specific_dissipation_rate      = specific_dissipation_rate


        ti['time_step'] = dt
        # ti['termination_step_count'] = int(np.max(t)/dt)
        yml.save(default_yaml_file)
        print('Main yaml: ', default_yaml_file)

        # --- Use nalu_aseq to generate all meshes and input files
        nalu_files, nalu_batches = nalu_aseq(input_file=default_yaml_file,
                  aseq=aseq, 
                  batch_file=batch_template,
                  jobname=jobname+'_',
                  submit=submit,
                  one_job=one_job,
                  raiseError=False, # <<<<<<<<<<<<
                  hours=hours, nodes=nodes, ntasks=ntasks, mem=mem, cluster=cluster)
                  #sim_dir = sim_dir,

        All_Batches+=nalu_batches
        All_Inputs+=nalu_files

        # --- Write a batch file with all
        sbatch_file = os.path.join(sim_dir, '_sbatch_all.sh')
        with open(sbatch_file, 'w', newline="\n") as f:
            for b in nalu_batches:
                bb = os.path.relpath(b, sim_dir)
                prefix='sbatch ' if cluster!='local' else ''
                f.write(f'{prefix}{bb}\n')
        print('SBatch:    ', sbatch_file)


# --- 
sbatch_file_all='_bashes_'+os.path.basename(__file__).replace('.py', '_list.sh')
with open(sbatch_file_all, 'w', newline="\n") as f:
    for b in All_Batches:
        prefix='sbatch ' if cluster!='local' else ''
        f.write(f'{prefix}{b}\n')
print('SBatch all:', sbatch_file_all)

