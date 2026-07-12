""" 
Generate PNG and CSV for each polars

"""

import os
import numpy as np
import glob
import matplotlib.pyplot as plt

from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_forces import polar_postpro, standardize_polar_df, plot_polars
from nalulib.nalu_forces_combine import nalu_forces_combine
from nalulib.weio.csv_file import CSVFile
from welib.tools.strings import FAIL, WARN, OK, INFO
from helper_functions import airfoil2configStat

# --- TORQUE
suffix='_nawea'
# airfoil_names = db.configs['airfoil'].unique()
# db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
# # db = db.select({'Roughness':'Clean'})
# db = db.query('airfoil!="L303"') # No geometry for L303
# db_stat = DataFrameDatabase('experiments/glasgow/DB_exp_static.pkl')
# # db_stat = db_stat.select({'Roughness':'Clean'})
# db_stat = db_stat.query('airfoil!="L303"') # No geometry for L303
# db_stat2 = DataFrameDatabase('./experiments/DB_misc_stat.pkl')

# --- NAWEA
suffix        = '_nawea'
out_dir       = '_results'+suffix
cases         = CSVFile('airfoils_data/DB_NAWEA_configs.csv').toDataFrame()
airfoil_names = cases['airfoil'].unique().tolist()
db            = DataFrameDatabase('airfoils_data/DB_iea_stat.pkl')
dbn           = DataFrameDatabase('airfoils_data/DB_NACA_stat.pkl')
db            = DataFrameDatabase.concatenate( (db, dbn))
print(db)



polout_dir  = os.path.join(out_dir,'_polars/')
figout_dir  = os.path.join(out_dir,'_figs/')
case_dir_2d = os.path.join(out_dir,'cases_polar2d'+suffix)


# airfoil_names =  list(airfoil_names) 
# airfoil_names =[]
# # airfoil_names = ['du00-w-212', 'nlf1-0416', 'ffa-w3-211'] + ['S809'] #  +  list(airfoil_names)
# # airfoil_names = ['du00-w-212']
# airfoil_names += ['S809']
# airfoil_names +=['du00-w-212', 'nlf1-0416', 'ffa-w3-211']
# airfoil_names = ['S809','NACA4415']
# airfoil_names = ['NACA4415']
# airfoil_names = ['S813']
# airfoil_names = ['LS-0421MOD']
# airfoil_names +=['du00-w-212']
# airfoil_names +=['nlf1-0416']
# airfoil_names +=['ffa-w3-211']
# airfoil_names +=['ffa-w3-301']
# airfoil_names +=['ffa-w3-241']
# airfoil_names +=['ffa-w3-360']
# airfoil_names +=['naca0018']


os.makedirs(polout_dir, exist_ok=True)
os.makedirs(figout_dir, exist_ok=True)

case_dir_n = {
        4:  os.path.join(out_dir, f'cases_polar3d{suffix}_z4_n4/'),
        24: os.path.join(out_dir, f'cases_polar3d{suffix}_z4_n24/'),
        121:os.path.join(out_dir, f'cases_polar3d{suffix}_z4_n121/'),
 }

# --- Loop through airfoils
for airfoil_name in airfoil_names:
    print(f'---------------------------- {airfoil_name} ------------------------')
#     config, db_stat_arf = airfoil2config(airfoil_name, db, db_stat, db_stat2)

    config, _ = airfoil2configStat(airfoil_name, cases)
    db_arf = db.select({'airfoil':airfoil_name})


    print('Reynolds:', config['Reynolds'].tolist())

    for re in config['Reynolds']:
        print(f'---------------------------- Re={re} ---')
        base = airfoil_name + '_re{:05.2f}M'.format(re)

        # --- Experimental polars
        print('--- POLARS EXP')
        polars={}
        
        if len(db_arf)==0:
            print('[WARN] Airfoil not in db')
        else:
            #db2 = db_stat_arf.select_approximate('Re', re, 0.101)
            db2 = db_arf.select_closest('Re', re)
            if len(db2.configs['Re'].unique())!=1:
                raise Exception('>>> Problem in select', re)
            if 'Setup' in db2.keys():
                polars = db2.toDict('Setup')
            elif 'Roughness' in db2.keys():
                polars = db2.toDict('Roughness')
                polars = {f"Exp {k}": v for k, v in polars.items()}
            # --- Export polars to CSV
            if len(polars)>0:
                print('Polars:   ', list(polars.keys()))
                for k,pol in polars.items():
                    if len(pol)>0:
                        pol = standardize_polar_df(pol)
                        suffix=k.replace(' ','_')
                        #print('TODO t11 make sure suffic Exp clean is replaced by EXP')
                        suffix = suffix.replace('Exp_Clean', 'EXP')
                        polfile = os.path.join(polout_dir, base+'_'+suffix+'.csv')
                        print('Writing:', polfile)
                        pol.to_csv(polfile, index=False) # <<<<<<<<<<<<<<<<<<< CSV EXPORT
                        pol = pol[(pol['Alpha'] >= -50) & (pol['Alpha'] <= 50)]
                        polars[k] = pol
# 
#         # --- CFD 3D polars 
#         print('--- POLARS 3D')
#         polars_3d={}
#         for n, case_dir_3d in case_dir_n.items():
#             sim_dir = os.path.join(case_dir_3d, base)
#             yaml_file3d = os.path.join(sim_dir,'input_aoa00.0.yaml')
#             polar_out3d = os.path.join(polout_dir, base+'_CFD3D_n{}.csv'.format(n)) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CSV EXPORT
#             pattern = os.path.join(sim_dir,'forces_pp*.csv')
#             input_files_pp = glob.glob(pattern)
#             if not os.path.exists(sim_dir):
#                 raise Exception(f'Folder not found, {sim_dir}')
#             try:
#                 pattern = os.path.join(sim_dir,'forces_aoa*.csv')
#                 dfp, dfss, _ = polar_postpro(pattern, yaml_file3d, polar_out = polar_out3d, use_ss=True, plot=False, verbose=False, span=4, cfd_ls='-', cfd_m='o')
#             except FileNotFoundError as e:
#                 FAIL('Not Combine Fail: FileNotFound', e)
#                 continue
#             polars_3d[f'CFD3D_n{n}'] = dfp
# 
#         polars = {**polars_3d, **polars}

        # --- CFD 2D polars 
        print('--- POLARS 2D')
        sim_dir = os.path.join(case_dir_2d, base)
        pattern = os.path.join(sim_dir,'forces*.csv')
        yaml_file2d = os.path.join(sim_dir,'input_aoa00.0.yaml')
        polar_out2d = os.path.join(polout_dir, base+'_CFD2D.csv') # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< CSV EXPORT
        try:
            dfp, dfss, _ = polar_postpro(pattern, yaml_file2d, polar_out = polar_out2d, use_ss=True, plot=False, verbose=False)
            polars = {'cfd': dfp, **polars}
        except FileNotFoundError as e:
            print('FileNotFoundError', e)



        print('--- PLOT')
        ylim = None
#         if airfoil_name =='S809':
        ylim  = [-1, 2]
        xlimCd= [0, 0.3]
        xlimAlpha= [-15, 30]
        fig = plot_polars(polars, verbose=True, ylim=ylim, plotCm=airfoil_name!='naca0018', xlimCd=xlimCd, xlimAlpha=xlimAlpha)
        fig.suptitle(base.replace('_',' '))
        figfile = os.path.join(figout_dir, base+'.png')
        fig.savefig(figfile)
        print('[INFO] Figure: ', figfile)
        plt.close(fig)

        #print(dfss)
        #print(dfp)




plt.show()
