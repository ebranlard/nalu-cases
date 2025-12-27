import os
import numpy as np
import glob
from nalulib import pyhyp
from nalulib.tools.dataframe_database import DataFrameDatabase

airfoil_dir ='airfoil_meshes'
mesh_dir    ='meshes'

# db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
# db = db.select({'Roughness':'Clean'})
# db = db.query('airfoil!="L303"') # No geometry for L303
# airfoil_names = db.configs['airfoil'].unique()

db = DataFrameDatabase('experiments/DB_all_stat.pkl')
db = db.query('airfoil!="L303"') # No geometry for L303
airfoil_names_db = db['airfoil'].unique()


airfoil_names = []
airfoil_names += ['S809'] 
# airfoil_names += airfoil_names_db
airfoil_names += ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']


l_res=600

# Reynolds = [0.1, 0.5, 0.75, 1, 2, 5, 10] # Milliions
#Reynolds = [1] # Milliions


# --- Grids from test cases, reproduced with pyhyp
# Background                       30 (range = 60)
# nlf1-0416     -n 145 --marchDist 1100  --s0 2.1e-6            -i nlf1-0416.csv       # Re = 4M 
# DU90-w2-225   -n 129 --marchDist 65    --s0 2.5e-6            -i du91-w2-225_nalu_l400.csv    # Re = 2M
# DU00-w-212    -n 145 --marchDist 105   --s0 2.8e-7            -i du00-w-212_re3M.csv  # Re = 3M
# DU00-w-212    -n 145 --marchDist 105   --re 3e6   --yplus 0.1 -i du00-w-212_re3M.csv  # Re= 3M , you need y+=0.035
# --- TODO
# FFA_Near_Body -n 271 --marchDist 25,   --s0 2.76e-6,  # Not reproduced yet
# --- My grids
# S809          -n 145 --marchDist 75	--re 2e6    --yplus 0.1 -i S809_l600.csv 
# S809          -n 145 --marchDist 75	--re 1e7    --yplus 0.1 -i S809_l600.csv 
# S809          -n 145 --marchDist 75	--re 1e6    --yplus 0.1 -i S809_l600.csv 
# S809          -n 145 --marchDist 75	--re 0.1e6  --yplus 0.1 -i S809_l600.csv 

N = 150
marchDist = 25
yplus=0.1

# create mesh dir
if not os.path.exists(mesh_dir):
    os.makedirs(mesh_dir)

# --- Loop through airfoils and create meshes
FAILED=[]
for airfoil_name in airfoil_names:
    print('----------------------------------------------------------------------')
    print(f'{airfoil_name:-^70}')
    print('----------------------------------------------------------------------')
    input_file = os.path.join(airfoil_dir, f'{airfoil_name}_l{l_res}.csv')
    print('Input file:', input_file)

    db_arf = db.select({'airfoil':airfoil_name})
    Reynolds = db_arf['Re'].round(2).sort_values().unique()
    print('Reynolds: ', Reynolds, '({})'.format(len(Reynolds)))

    for re in Reynolds:
        print(f'---------------------------- Re={re}')
        output_file = os.path.join(mesh_dir, f'{airfoil_name}_m{N}_n1_re{re:05.2f}M_y{yplus}mu.exo')
        if not os.path.exists(output_file):
            print(f'Creating mesh for {airfoil_name} with Re={re}M, N={N}, y+={yplus}')
            try:
                pyhyp(input_file=input_file, output_file=output_file, re=re*1e6, marchDist=marchDist, N=N, yplus=yplus, verbose=True)
            except:
                print(f'Failed to create mesh for {airfoil_name} with Re={re}M, N={N}, y+={yplus}')
                FAILED.append((airfoil_name, re, N, yplus))	
        else:
            print(f'[SKIP] {airfoil_name} {re}')

#         import pdb; pdb.set_trace()

for failed in FAILED:
	print(f'Failed to create mesh for {failed[0]} with Re={failed[1]}M, N={failed[2]}, y+={failed[3]}')
