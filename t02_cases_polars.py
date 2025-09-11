import os
import numpy as np
import glob
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_aseq import nalu_aseq
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus_rotate import exo_rotate
from nalulib.exodus_quads2hex import exo_zextrude



mesh_dir    ='meshes'
case_dir    ='cases_polar'

nalu_template ='_templates/airfoil_name/input_2d_rans_du00-w-212.yaml'
batch_template ='_templates/submit-kestrel_n1.sh'
submit=False
hours=2
nodes=1
ntasks=None
mem=None

aseq = np.arange(-20, 30+3/2, 1)
# aseq = np.arange(-2, 3+3/2, 1)


chord=1
density= 1.2
viscosity = 9.0e-06
# TODO TI
N = 150
yplus=0.1

# 
db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
db = db.select({'Roughness':'Clean'})
db = db.query('airfoil!="L303"') # No geometry for L303
airfoil_names = db.configs['airfoil'].unique()

airfoil_names =  list(airfoil_names) + ['du00-w2-212', 'nfl1-0416'] 
# airfoil_names = ['du00-w2-212'] #, 'nfl1-0416']  +  list(airfoil_names)



yml_template = NALUInputFile(nalu_template)

# def create_polar_case(nalu_template, re, mesh_file_2d, sim_dir='./', basename='', nSpan=4, density=1.2, viscosity=9.0e-6, chord=1, background_3d=None):
#     if mean_round is None:
#         mean_round = alpha_mean
#     if re_round is None:
#         re_round = re
# 
#     if isinstance(nalu_template, str):
#         yml_in = NALUInputFile(nalu_template)
#     else:
#         yml_in = nalu_template
# 
#     sim_dir = os.path.join(case_dir, airfoil_name)
#     local_mesh_dir = os.path.join(sim_dir, 'meshes')
#     if not os.path.exists(sim_dir):
#         os.makedirs(sim_dir)
#     if not os.path.exists(local_mesh_dir):
#         os.makedirs(local_mesh_dir)
# 
#     U = float(re*1e6 *viscosity /(density * chord ))
#     dt = float(np.around(0.02 * chord / U, 8))
#     T = float(1/frequency)
# 
#     T_steady = chord/U*nT_steady
# 
# 
#     basename_ReMean = basename+'_'+'re{:04.1f}_mean{:04.1f}'.format(re_round, mean_round)
#     basename = basename_ReMean+'_'+'A{:04.1f}_f{:03.1f}'.format(amplitude, frequency)
#     yaml_file = os.path.join(sim_dir, basename+'.yaml')
# 
# 
#     # --- Creating meshes
#     rotated_mesh_2d  = os.path.join(local_mesh_dir, basename_ReMean+'_n1.exo')
#     if not os.path.exists(rotated_mesh_2d):
#         print('Rotating mesh: ', rotated_mesh_2d, alpha_mean)
#         exo_rotate(mesh_file_2d, rotated_mesh_2d, angle=alpha_mean, center=(0,0), angle_center=None, 
#                       inlet_start=None, inlet_span=None, outlet_start=None, keep_io_side_set=False, 
#                       inlet_name='inlet', outlet_name='outlet',
#                       verbose=False, profiler=False, debug=False)
# 
#     # --- Change yaml file
#     yml = yml_in.copy()
#     # Shortcuts 
#     ti = yml.data['Time_Integrators'][0]['StandardTimeIntegrator']
#     realms = yml.data['realms']
# 
#     bg = realms[0]
#     af = realms[1]
#     bg['mesh'] = os.path.relpath(background_3d, sim_dir).replace('\\', '/')
#     af['mesh'] = os.path.relpath(extruded_mesh_2d, sim_dir).replace('\\', '/')
# 
#     #if 'restart' in bg:
#     #    bg['restart']['restart_data_base_name'] = 'restart/'+basename_ReMean+'_bg'
#     #    af['restart']['restart_data_base_name'] = 'restart/'+basename_ReMean+'_arf'
# 
# 
#     if not os.path.exists(os.path.join(sim_dir,'forces')):
#         os.makedirs(os.path.join(sim_dir,'forces'))
# 
#     pp = af['post_processing'][0]['output_file_name'] = 'forces/'+basename+'_pp.csv'
#     pp = af['post_processing'][1]['output_file_name'] = 'forces/'+basename+'.csv'
# 
# 
#     yml.velocity = [U, 0, 0]
#     yml.density = density
#     yml.viscosity = viscosity
#     ti['time_step'] = dt
# 
#     t_steady = np.max([2*T, 100*dt, T_steady])
#     t, x, y, theta = yml.set_sine_motion(A=amplitude, f=frequency, n_periods=nT, t_steady=t_steady, dt=dt, DOF='pitch', irealm=1)
# 
#     ti['termination_step_count'] = int(np.max(t)/dt)
# 
# 
#     if batch_template is not None:
#         batch_file = nalu_batch(batch_template, nalu_input_file=yaml_file, jobname='p'+basename, sim_dir=sim_dir, mail=True)
#     else:
#         batch_file =None
# 
#     yml.save(yaml_file)
#     return yaml_file, batch_file



# --- Loop through airfoils and create meshes
for airfoil_name in airfoil_names:
    print(f'---------------------------- {airfoil_name} ------------------------')
    db_arf = db.select({'airfoil':airfoil_name})

    Reynolds = db_arf.configs['Re'].round(1).unique()

    # --- HACK ['du00-w2-212', 'nfl1-0416', 'ffa']:
    hack=False
    if len(db_arf)==0:
        hack=True
        if airfoil_name == 'du00-w2-212':
            mesh_file_2d = './du00-w-212/grids/du00w212_re3M_y03_aoa0_n1.exo'
            Reynolds=[3]
        elif airfoil_name == 'nfl1-0416':
            mesh_file_2d = './nfl1-0416/grids/nlf1-0416_re4M_y2_aoa0_n1.exo'
            Reynolds=[4]
        elif airfoil_name == 'ffa':
            mesh_file_2d = './ffa/grids/ffa_w3_211_near_body_aoa0_n1.exo'
            Reynolds=[10]
        else:
            raise NotImplementedError(airfoil_name)

    Reynolds.sort()
    Reynolds = Reynolds[-1::-1]
    print('Reynolds: ', Reynolds, '({})'.format(len(Reynolds)))

    for re in Reynolds:

        # TODO TODO TODO TO REMEMBER NEAR BODY!
        if hack:
            pass
        else:
            mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re}M_y{yplus}mu.exo')

        if not os.path.exists(mesh_file_2d):
            print('[WARN] Mesh not found: ', mesh_file_2d)
            continue

        jobname = airfoil_name + '_re{:02.1f}M'.format(re)
        sim_dir = os.path.join(case_dir, airfoil_name + '_re{:02.1f}M'.format(re))
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
        dt = float(np.around(0.02 * chord / U, 8))
        yml.velocity = [U, 0, 0]
        yml.density = density
        yml.viscosity = viscosity
        ti['time_step'] = dt
        # ti['termination_step_count'] = int(np.max(t)/dt)


        yml.save(default_yaml_file)

        # --- Use nalu_aseq to generate all meshes and input files
        nalu_aseq(input_file=default_yaml_file,
                  aseq=aseq, 
                  batch_file=batch_template,
                  jobname=jobname+'_',
                  submit=submit,
                  #sim_dir = sim_dir,
                  hours=hours, nodes=nodes, ntasks=ntasks, mem=mem)

        break
