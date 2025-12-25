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
nT_steady=60

nSpan = 121
# nSpan = 24
# nSpan = 4
for nSpan in [121]:
    aseq = np.arange(-5, 20+3/2, 5)
    aseq = np.arange(-5, 25+3/2, 2.5)
    #aseq = np.arange(-20, 25+3/2, 5)
    # aseq = np.arange(-2, 3+3/2, 1)
    one_job = False


    mesh_dir    ='meshes'
    case_dir    ='cases_polar3d_n{}'.format(nSpan)
    nalu_template ='_templates/airfoil_name/input_no_restart_with_output.yaml'
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
        batch_template ='_templates/submit-kestrel.sh'
        hours={4:3, 24:8, 121:48}[nSpan]
        nodes={4:1, 24:1, 121:1}[nSpan]
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
    # airfoil_names = ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']  +  list(airfoil_names)
    airfoil_names = ['S809']
    #airfoil_names += ['du00-w-212', 'ffa-w3-211', 'nlf1-0416']
    airfoil_names = ['du00-w-212', 'nlf1-0416']
    #airfoil_names = ['nlf1-0416']

    print(f'-------------------------------- SETUP ---------------------------------')
    print(f'cluster      : {cluster}')
    print(f'hours        : {hours}')
    print(f'ntasks       : {ntasks}')
    print(f'airfoil_names: {airfoil_names}')



    background_3d = './meshes/background_n{}.exo'.format(nSpan)

    if not os.path.exists(background_3d):
        background_3d_n1 = './meshes/background_n1.exo'
        exo_zextrude(background_3d_n1, background_3d, nSpan=nSpan, zSpan=4.0, zoffset=0.0, verbose=True, airfoil2wing=False, ss_wing_pp=False, profiler=False, ss_suffix='_bg')






    yml_template = NALUInputFile(nalu_template)

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


        for iRe, re in enumerate(Reynolds):

            # TODO TODO TODO TO REMEMBER NEAR BODY!
            if hack:
                pass
            else:
                mesh_file_2d = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:04.1f}M_y{yplus}mu.exo')

            if not os.path.exists(mesh_file_2d):
                print('[WARN] Mesh not found: ', mesh_file_2d)
                continue

            jobname = airfoil_name + '_re{:04.1f}M'.format(re)
            sim_dir = os.path.join(case_dir, jobname)
            print('sim_dir:   ', sim_dir)
            if not os.path.exists(sim_dir):
                os.makedirs(sim_dir)
            local_mesh_dir = os.path.join(sim_dir, 'meshes')
            if not os.path.exists(local_mesh_dir):
                os.makedirs(local_mesh_dir)


            # --- Scales
            U = float(re*1e6 *viscosity /(density * chord ))
            dt = float(np.around(dt_fact * chord / U, 8))
            T = chord/U*nT_steady


            # --- Creating meshes
            extruded_mesh = os.path.join(local_mesh_dir, 'input_mesh'+'_n{}.exo'.format(nSpan))
            if not os.path.exists(extruded_mesh):
                exo_zextrude(mesh_file_2d, extruded_mesh, nSpan=nSpan, zSpan=4.0, zoffset=0.0, verbose=False, airfoil2wing=True, ss_wing_pp=True, profiler=False, ss_suffix=None)

            #mesh_file_2d_rel = os.path.relpath(mesh_file_2d, sim_dir)

            # --- Create a input file with proper mesh and flow parameters
            default_yaml_file = os.path.join(sim_dir, 'input.yaml')

            yml = yml_template.copy()
            # Shortcuts 
            ti = yml.data['Time_Integrators'][0]['StandardTimeIntegrator']
            realms = yml.data['realms']
            bg = realms[0]
            af = realms[1]
            bg['mesh'] = os.path.relpath(background_3d, sim_dir).replace('\\', '/')
            af['mesh'] = os.path.relpath(extruded_mesh, sim_dir).replace('\\', '/')
            if 'restart' in bg:
                bg['restart']['restart_data_base_name'] = 'restart/'+jobname+'_bg'
                af['restart']['restart_data_base_name'] = 'restart/'+jobname+'_arf'

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

            # Time
            ti['time_step'] = dt
            ti['termination_step_count'] = int(T/dt)


            # --- Output
            if 'output' in bg:
                #bg['output']['output_data_base_name'] # handled by polar_aseq
                bg['output']['output_frequency'] = int(T/dt)-1
                af['output']['output_frequency'] = int(T/dt)-1

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
                      #sim_dir = sim_dir,
    # 
    #         for ia, alpha_mean in enumerate(Alpha_mean):
    #             for iA, amplitude in enumerate(Amplitudes):
    #                 yml, batch = create_step_case(alpha_mean, amplitude, nT_steady, re, mesh_file_2d, background_3d, yml, sim_dir, basename=airfoil_name, nSpan=nSpan, density=density, viscosity=viscosity, batch_template=batch_template, nramp=nramp)
    #                 print('[YML]', yml)
    #                 print('[BAT]', batch)

            # --- Write a batch file with all
            sbatch_file = os.path.join(sim_dir, '_sbatch_all.sh')
            with open(sbatch_file, 'w', newline="\n") as f:
                for b in nalu_batches:
                    bb = os.path.relpath(b, sim_dir)
                    prefix='sbatch ' if cluster!='local' else ''
                    f.write(f'{prefix}{bb}\n')
            print('SBatch:    ', sbatch_file)

            if iRe==0:
                print('[WARN] breaking after first Re')
                break

    #     if ia==0:
    #         break

