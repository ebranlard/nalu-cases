""" 
Loads CSV files ith coordinates of Glasgow experiment airfoils.
Remesh them to 600 points

equivalent to running:

    arf-mesh -n 301 --a_hyp 3 --respline file.csv -o file_l600.csv 

"""

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from nalulib.airfoil_mesher import mesh_airfoil # arf_mesh
from nalulib.weio.csv_file import CSVFile


# --- Remeshing inputs:
# NOTE:
#  - FFA_211 from Shreyas have 525 points
#  - DU00-w-212 have 782 points
#  - DU91 has 385 points
LL=500
n = int(LL/2)+1
a_hyp = 3


# --- Selecting airfoil names
# --- TORQUE 
# airfoil_dir     ='airfoils_data/coords_raw_torque/'
# airfoil_out_dir ='airfoils_data/coords_meshed_torque'
# suffix=''
# TE_TYPE=None
#

# --- NAWEA 
airfoil_dir     ='airfoils_data/coords_raw_nawea/'
airfoil_out_dir ='airfoils_data/coords_meshed_nawea/'
cases = CSVFile('airfoils_data/DB_NAWEA_configs.csv').toDataFrame()
airfoil_names = cases['airfoil'].unique().tolist()
suffix='_coords'
TE_TYPE=None #, 'sharp' # TODO need to implement an equispacing based on last spacing of upper or lower surface

#airfoil_names=['ffa-w3-211']
#airfoil_names=['naca4430']
#airfoil_names=['snl-ffa-w3-420fb']


# --- Loop on airfoils, and resample the data
if not os.path.exists(airfoil_out_dir):
    os.makedirs(airfoil_out_dir)

for ia, arf_name in enumerate(airfoil_names):
    print(f'---------------------- {arf_name} --------------------------')
    # Find the CSV file for the current airfoil
    #csv_file_in = os.path.join('airfoils', f'{arf_name}.csv')
    csv_file_in  = os.path.join(airfoil_dir    , f'{arf_name}{suffix}.csv')
    csv_file_out = os.path.join(airfoil_out_dir, f'{arf_name}_l{LL}.csv')
    fig_file_out = os.path.join(airfoil_out_dir, f'{arf_name}_l{LL}.png')

    arf = mesh_airfoil(csv_file_in, output_file=csv_file_out, n=n, respline=True, method_te='min_dist', TE_type=TE_TYPE, a_hyp=a_hyp, plot=False, verbose=False)

    print('>>> Tri plot')
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

    #if ia==0:
    #    break


plt.show()
print('Done resampling airfoils.')

