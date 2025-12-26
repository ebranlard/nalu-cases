import os
import numpy as np
import glob
import matplotlib.pyplot as plt

from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_forces import polar_postpro, standardize_polar_df, plot_polars
from nalulib.nalu_forces_combine import nalu_forces_combine
from nalulib.weio.csv_file import CSVFile

def airfoil2config(airfoil_name, db, db_stat=None, db_stat2=None):
    config={}
    config['chord']                    = 1
    config['density']                  = 1.2
    config['viscosity']                = 9.0e-06
    config['specific_dissipation_rate']= 114.54981120000002
    config['turbulent_ke']             = 0.0013020495206400003
    config['dt_fact']                  = 0.02

    if airfoil_name in ['du00-w-212', 'nlf1-0416' , 'ffa-w3-211']:
        db_stat_arf = db_stat2.select({'airfoil':airfoil_name})
        config['Reynolds'] = np.array(sorted(db_stat_arf.configs['Re'].round(2).unique()))
        config_ = db_stat_arf.common_config
        for k,v in config_.items():
            if isinstance(v, (int, float)):
                if not np.isnan(v):
                    print(f'Setting {k:20s}={v}')
                    config[k] = v
            else:
                config[k] = v
    else:
        db_arf = db.select({'airfoil':airfoil_name})
        if len(db_arf) ==0:
            raise Exception('Something is wrong')
        db_stat_arf = db_stat.select({'airfoil':airfoil_name})
        if airfoil_name=='S809':
            Re1 = np.array(sorted(db_arf.configs['Re'].round(1).unique()))
            config['Reynolds'] =Re1
        else:
            Re1 = np.array(sorted(db_arf.configs['Re'].round(2).unique()))
            RE_expected = np.array([0.8, 1.0, 1.2, 1.4, 1.5]) # 0.75, 1.0, 1.25, 1.3, 1.4, 1.5]
            RE = []
            for re in Re1:
                i=np.argmin(np.abs(re- RE_expected))
                re_ = RE_expected[i]
                RE.append(re_)
            RE=np.array(sorted(list(set(RE))))
            config['Reynolds'] =RE

        config['db_Stat'] = db_stat_arf

    return config, db_stat_arf

# --- Main inputs
out_dir = '_results'
polout_dir = '_results/_polars/'
figout_dir = '_results/_figs/'
case_dir_2d = '_results/cases_polar'

db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
# db = db.select({'Roughness':'Clean'})
db = db.query('airfoil!="L303"') # No geometry for L303
airfoil_names = db.configs['airfoil'].unique()


db_stat = DataFrameDatabase('experiments/glasgow/DB_exp_static.pkl')
# db_stat = db_stat.select({'Roughness':'Clean'})
db_stat = db_stat.query('airfoil!="L303"') # No geometry for L303
db_stat2 = DataFrameDatabase('./experiments/DB_misc_stat.pkl')



airfoil_names =  list(airfoil_names) 
airfoil_names = ['du00-w-212', 'nlf1-0416', 'ffa-w3-211'] + ['S809'] #  +  list(airfoil_names)
# airfoil_names = ['du00-w-212']
airfoil_names = ['S809']
airfoil_names +=['du00-w-212', 'nlf1-0416', 'ffa-w3-211']
# airfoil_names = ['S809','NACA4415']
# airfoil_names = ['NACA4415']
# airfoil_names = ['S813']
# airfoil_names = ['LS-0421MOD']
# airfoil_names =['ffa-w3-211']
# airfoil_names =['du00-w-212']
# airfoil_names =['nlf1-0416']



os.makedirs(polout_dir, exist_ok=True)
os.makedirs(figout_dir, exist_ok=True)

case_dir_n = {
        4:'_results/cases_polar3d_n4/',
        24:'_results/cases_polar3d_n24/',
        121:'_results/cases_polar3d_n121/',
 }

# --- Loop through airfoils
for airfoil_name in airfoil_names:
    print(f'---------------------------- {airfoil_name} ------------------------')
    config, db_stat_arf = airfoil2config(airfoil_name, db, db_stat, db_stat2)
    print('Reynolds:', config['Reynolds'].tolist())

    # HACK
    if len(config['Reynolds'])>2:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HACK ONE RE FOR NOW')
        config['Reynolds'] = [0.8]

    for re in config['Reynolds']:
    #for re in [0.8]:
        print(f'---------------------------- Re={re} ---')
        base = airfoil_name + '_re{:04.1f}M'.format(re)

        # --- Experimental polars
        print('--- POLARS EXP')
        polars={}
        #db2 = db_stat_arf.select_approximate('Re', re, 0.101)
        db2 = db_stat_arf.select_closest('Re', re)
        if len(db2.configs['Re'].unique())!=1:
            print('>>> Problem in select', re)
        if 'Setup' in db2.keys():
            polars = db2.toDict('Setup')
        elif 'Roughness' in db2.keys():
            polars = db2.toDict('Roughness')
            polars = {f"Exp {k}": v for k, v in polars.items()}
        if len(polars)>0:
            print('Polars:   ', list(polars.keys()))
            for k,pol in polars.items():
                if len(pol)>0:
                    pol = standardize_polar_df(pol)
                    pol.to_csv(os.path.join(polout_dir, base+'_'+k.replace(' ','_')+'.csv'), index=False)

        # --- CFD 3D polars 
        print('--- POLARS 3D')
        polars_3d={}
        combine=True
        for n, case_dir_3d in case_dir_n.items():
            sim_dir = os.path.join(case_dir_3d, base)
            yaml_file3d = os.path.join(sim_dir,'input_aoa00.0.yaml')
            polar_out3d = os.path.join(polout_dir, base+'_CFD3D_n{}.csv'.format(n))
            if combine:
                try:
                    pattern = os.path.join(sim_dir,'forces_*.csv')
                    csv_files = nalu_forces_combine(pattern=pattern, dry_run=False, verbose=False)
                    pattern = os.path.join(sim_dir,'_forces_aoa*.csv')
                    dfp, dfss, _ = polar_postpro(pattern, yaml_file3d, polar_out = polar_out3d, use_ss=True, plot=False, verbose=False, span=4, cfd_ls='-', cfd_m='o')
                except FileNotFoundError as e:
                    #print('FileNotFound', e)
                    continue
            else:
                try:
                    pattern = os.path.join(sim_dir,'_forces_aoa*.csv')
                    dfp, dfss, _ = polar_postpro(pattern, yaml_file3d, polar_out = polar_out3d, use_ss=True, plot=False, verbose=False, span=4, cfd_ls='-', cfd_m='o')
                except FileNotFoundError as e:
                    print('FileNotFound', e)
                    continue
            polars_3d[f'CFD3D_n{n}'] = dfp

        polars = {**polars_3d, **polars}

        # --- CFD 2D polars 
        print('--- POLARS 2D')
        sim_dir = os.path.join(case_dir_2d, base)
        pattern = os.path.join(sim_dir,'forces*.csv')
        yaml_file2d = os.path.join(sim_dir,'input_aoa00.0.yaml')
        polar_out2d = os.path.join(polout_dir, base+'_CFD2D.csv')
        try:
            dfp, dfss, _ = polar_postpro(pattern, yaml_file2d, polar_out = polar_out2d, use_ss=True, plot=False, verbose=False)
            polars = {'cfd': dfp, **polars}
        except FileNotFoundError as e:
            print('FileNotFoundError', e)

        


        print('--- PLOT')
        fig = plot_polars(polars, verbose=True)
        fig.suptitle(base.replace('_',' '))
        figfile = os.path.join(figout_dir, base+'.png')
        fig.savefig(figfile)
        print('[INFO] Figure: ', figfile)
        plt.close(fig)

        #print(dfss)
        #print(dfp)




plt.show()
