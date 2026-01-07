import numpy as np

# import os
# import numpy as np
# import glob
# from nalulib import pyhyp
# from nalulib.tools.dataframe_database import DataFrameDatabase
import matplotlib.pyplot as plt
from nalulib import mesh_airfoil # arf_mesh
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.airfoil_shapes import StandardizedAirfoilShape

from welib.essentials import *
from welib.tools.figure import setFigureTitle

setFigurePath('../nalu_torque_2026/figs/')
setFigureFont(12)
# setFigureTitle(1)
# export=False
# export=True

airfoil_names = ['S809','du00-w-212', 'nlf1-0416']


fig, axes = plt.subplots( 3, 3, gridspec_kw={'width_ratios': [1, 5, 1]}, figsize=(12.8, 3.9))
fig.subplots_adjust(left=0.04, right=0.99, top=0.999, bottom=0.001, hspace=0.04, wspace=0.20)

for i, arf_name in enumerate(airfoil_names):
    print(f'---------------------- {arf_name} --------------------------')
    # Find the CSV file for the current airfoil
    filename = f'./airfoil_meshes/{arf_name}_l600.csv'

    arf = StandardizedAirfoilShape(filename=filename)

#     if arf_name in ['du00-w-212', 'nlf1-0416', 'ffa-w3-211']:
#         csv_file_in = os.path.join('airfoils', f'{arf_name}.csv')
#     else:
#         csv_file_in = os.path.join(airfoil_dir, f'{arf_name}.csv')
#     #fig_file_out = os.path.join(airfoil_out_dir, f'{arf_name}_l{l}.png')
# 
#     arf = mesh_airfoil(csv_file_in, n=n, respline=True, te_type=None, a_hyp=3, plot=False, verbose=False)
# 
    axes = np.asarray(axes)
    arf.tri_plot(     title=None, axes=axes[i,:], n_target=50, legend=False)
# 
#     axes[0,0].set_xlim(axes[1,0].get_xlim())
#     axes[0,0].set_ylim(axes[1,0].get_ylim())
# 
#     axes[0,2].set_xlim(axes[1,2].get_xlim())
#     axes[0,2].set_ylim(axes[1,2].get_ylim())
# 
#     fig.savefig(fig_file_out)
#     print('Export: ', csv_file_out)
#     print('Export: ', fig_file_out)
for ax in np.array(axes).flatten():
    ax.axis('off')

for i,ax in enumerate(np.array(axes)[:,1]):
    ax.set_aspect('equal', 'box')
    ax.set_ylim([-0.115, 0.125])
    ax.set_xlim([-0.1, 1.1])
    ax.text(0.40, 0, airfoil_names[i], ha='center', va='center' )
axes[1,0].text(0.01, 0, 'Leading edge', ha='center', va='center' )
axes[1,2].text(0.99,  0.0005, 'Trailing edge', ha='center', va='center' )

fig._title = 'airfoil_shapes'
export2pdf()

plt.show()
