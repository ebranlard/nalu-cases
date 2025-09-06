import os
import numpy as np
import glob
from nalulib import pyhyp
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus_rotate import exo_rotate
from nalulib.exodus_quads2hex import exo_zextrude



airfoil_dir ='airfoil_meshes'
mesh_dir    ='meshes'
case_dir    ='cases'
nalu_template ='_templates/airfoil_name/input.yaml'
batch_template ='_templates/submit-kestrel.sh'
density= 1.2
viscosity = 9.0e-06
background_2d = './meshes/background_n1.exo'
nSpan = 4
N = 150
yplus=0.1

# 
db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
db = db.select({'Roughness':'Clean'})
db = db.query('airfoil!="L303"') # No geometry for L303
airfoil_names = db.configs['airfoil'].unique()

background_3d = './meshes/background_n{}.exo'.format(nSpan)



yml = NALUInputFile(nalu_template)


def create_pitching_case(alpha_mean, amplitude, frequency, re, mesh_file_2d, background_3d, nalu_template, sim_dir, basename, nSpan=4, density=1.2, viscosity=9.0e-6, chord=1, mean_round=None, re_round=None, batch_template=None):
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

    U = float(re*1e6 *viscosity /(density * chord ))
    dt = float(np.around(0.02 * chord / U, 8))
    T = float(1/frequency)
    stepsPerT = int(T/dt)


    basename_ReMean = basename+'_'+'re{:04.1f}_mean{:04.1f}'.format(re_round, mean_round)
    basename = basename_ReMean+'_'+'A{:04.1f}_f{:03.1f}'.format(amplitude, frequency)
    yaml_file = os.path.join(sim_dir, basename+'.yaml')


    # --- Creating meshes
    rotated_mesh_2d  = os.path.join(local_mesh_dir, basename_ReMean+'_n1.exo')
    extruded_mesh_2d = os.path.join(local_mesh_dir, basename_ReMean+'_n{}.exo'.format(nSpan))
    if not os.path.exists(extruded_mesh_2d):
        if not os.path.exists(rotated_mesh_2d):
            print('Rotating mesh: ', rotated_mesh_2d, alpha_mean)
            exo_rotate(mesh_file_2d, rotated_mesh_2d, angle=alpha_mean, center=(0,0), angle_center=None, 
                          inlet_start=None, inlet_span=None, outlet_start=None, keep_io_side_set=False, 
                          inlet_name='inlet', outlet_name='outlet',
                          verbose=False, profiler=False, debug=False)
        print('Extrudingmesh: ', extruded_mesh_2d, nSpan)
        exo_zextrude(rotated_mesh_2d, extruded_mesh_2d, nSpan=nSpan, zSpan=4.0, zoffset=0.0, verbose=False, airfoil2wing=True, ss_wing_pp=True, profiler=False, ss_suffix=None)
        try:
            os.remove(rotated_mesh_2d)
        except:
            print('[WARN] Cant delete: ', rotated_mesh_2d)

    



    # --- Change yaml file

    yml = yml_in.copy()
    # Shortcuts 
    ti = yml.data['Time_Integrators'][0]['StandardTimeIntegrator']
    realms = yml.data['realms']

    bg = realms[0]
    af = realms[1]
    bg['mesh'] = os.path.relpath(background_3d, sim_dir).replace('\\', '/')
    af['mesh'] = os.path.relpath(extruded_mesh_2d, sim_dir).replace('\\', '/')

    bg['restart']['restart_data_base_name'] = 'restart/'+basename_ReMean+'_bg'
    af['restart']['restart_data_base_name'] = 'restart/'+basename_ReMean+'_arf'



    pp = af['post_processing'][0]['output_file_name'] = 'forces/'+basename+'_pp.csv'
    pp = af['post_processing'][1]['output_file_name'] = 'forces/'+basename+'.csv'


    yml.velocity = [U, 0, 0]
    yml.density = density
    yml.viscosity = viscosity
    ti['time_step'] = dt

    t_steady = max(2*T, 100*dt)
    t, x, y, theta = yml.set_sine_motion(A=amplitude, f=frequency, n_periods=10, t_steady=t_steady, dt=dt, DOF='pitch', irealm=1)

    ti['termination_step_count'] = int(np.max(t)/dt)


    if batch_template is not None:
        batch_file = nalu_batch(batch_template, nalu_input_file=yaml_file, jobname=basename, sim_dir=sim_dir)
    else:
        batch_file =None

    yml.save(yaml_file)
    return yaml_file, batch_file

# --- Loop through airfoils and create meshes
for airfoil_name in airfoil_names:
    db_arf = db.select({'airfoil':airfoil_name})
    Reynolds = db_arf.configs['Re'].round(1).unique()
    print('Reynolds: ', Reynolds, '({})'.format(len(Reynolds)))
    sim_dir = os.path.join(case_dir, airfoil_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    for re in Reynolds:
        mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re}M_y{yplus}mu.exo')
        for idx, config in db_arf.configs.iterrows():
            alpha_mean = config['mean_real']
            mean_round = config['Mean']
            amplitude = config['Amplitude']
            freq = config['Frequency']
            re_real = config['Re']
            print(dict(config))
            yml, batch = create_pitching_case(alpha_mean, amplitude, freq, re_real, mesh_file_2d, background_3d, yml, sim_dir, basename=airfoil_name, nSpan=nSpan, density=density, viscosity=viscosity, mean_round=mean_round, re_round=re, batch_template=batch_template)
            print('[YML]', yml)
            print('[BAT]', batch)
            if idx==0:
                break

