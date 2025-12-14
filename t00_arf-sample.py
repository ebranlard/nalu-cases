""" 
Loads CSV files ith coordinates of Glasgow experiment airfoils.
Remesh them to 600 points
"""

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from nalulib import mesh_airfoil # arf_mesh
from nalulib.tools.dataframe_database import DataFrameDatabase


# --- Remeshing inputs:
# NOTE:
#  - FFA_211 from Shreyas have 525 points
#  - DU00-w-212 have 782 points
#  - DU91 has 385 points
l = 600
n = int(l/2)+1
a_hyp = 3


# Selecting airfoil names
airfoil_dir ='experiments/glasgow/'
airfoil_out_dir ='airfoil_meshes'
db = DataFrameDatabase('experiments/glasgow/DB_exp_loop.pkl')
db = db.select({'Roughness':'Clean'})
db = db.query('airfoil!="L303"') # No geometry for L303
airfoil_names = ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']  +  list(airfoil_names)


# Loop on airfoils, and resample the data

if not os.path.exists(airfoil_out_dir):
    os.makedirs(airfoil_out_dir)

for arf_name in airfoil_names:
    print(f'---------------------- {arf_name} --------------------------')
    # Find the CSV file for the current airfoil
    if arf_name in ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']:
        csv_file_in = os.path.join('airfoils', f'{arf_name}.csv')
    else:
        csv_file_in = os.path.join(airfoil_dir, f'{arf_name}.csv')
    csv_file_out = os.path.join(airfoil_out_dir, f'{arf_name}_l{l}.csv')
    fig_file_out = os.path.join(airfoil_out_dir, f'{arf_name}_l{l}.png')

    arf = mesh_airfoil(csv_file_in, output_file=csv_file_out, n=n, respline=True, te_type=None, a_hyp=a_hyp, plot=False, verbose=False)


    fig, axes = plt.subplots( 2, 3, gridspec_kw={'width_ratios': [1, 5, 1]}, figsize=(18, 5.5))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.07, hspace=0.20, wspace=0.20)
    axes = np.asarray(axes)

    arf._ori.tri_plot(title=arf_name + '- Original', axes=axes[0,:], n_target=50, legend=True)
    arf.tri_plot(     title=arf_name + '- Remeshed', axes=axes[1,:], n_target=50, legend=True)

    axes[0,0].set_xlim(axes[1,0].get_xlim())
    axes[0,0].set_ylim(axes[1,0].get_ylim())

    axes[0,2].set_xlim(axes[1,2].get_xlim())
    axes[0,2].set_ylim(axes[1,2].get_ylim())

    fig.savefig(fig_file_out)
    print('Export: ', csv_file_out)
    print('Export: ', fig_file_out)


print('Done resampling airfoils.')

