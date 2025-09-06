import os
import numpy as np
import glob
from nalulib import mesh_airfoil # arf_mesh
from nalulib.tools.dataframe_database import DataFrameDatabase

airfoil_dir ='experiments/glasgow/'
airfoil_out_dir ='airfoil_meshes'
#'L303_MISSING.csv
#'S824_MISSING.csv,
# airfoil_names=[
# 'LS-0412MOD',
# 'LS-0417MOD',
# 'NACA4415',
# 'S801',
# 'S809',
# 'S810',
# 'S812',
# 'S813',
# 'S814',
# 'S815',
# 'S825',
# ]
db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
db = db.select({'Rougthness':'clean'})
db = db.query('airfoil!="L303"') # No geometry for L303
airfoil_names = db.configs['airfoil'].unique()

# import pdb; pdb.set_trace()


# Loop on airfoils, and resample the data
l = 600
n = int(l/2)+1
a_hyp = 3

if not os.path.exists(airfoil_out_dir):
    os.makedirs(airfoil_out_dir)

for arf in airfoil_names:
    print(f'---------------------- {arf} --------------------------')
    # Find the CSV file for the current airfoil
    csv_file_in = os.path.join(airfoil_dir, f'{arf}.csv')
    csv_file_out = os.path.join(airfoil_out_dir, f'{arf}_l{l}.csv')

    mesh_airfoil(csv_file_in, output_file=csv_file_out, n=l, respline=True, te_type=None, a_hyp=a_hyp, plot=False, verbose=False)
    


print('Done resampling airfoils.')
