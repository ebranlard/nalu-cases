import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import welib.weio as weio

from nalulib.nalu_forces import standardize_polar_df, plot_polar_df
from welib.essentials import *
from welib.tools.figure import setFigureTitle

setFigurePath('../nalu_torque_2026/figs/')
setFigureFont(14)
setFigureTitle(1)
export=False
export=True


def plot_polars(airfoil_name, Re):
    fig,axes = plt.subplots(1, 2, sharey=True, figsize=(4.8,3.0))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.92, bottom=0.17, hspace=0.20, wspace=0.08)
    plot_polar_df(axes , dfe , marker='o' , label='Exp'       , c='k'       , ls=''   , ms=3  , CdWithCl=False)
    plot_polar_df(axes , df1 , marker=''  , label='CFD 2D'    , c='k'       , ls='-'  , lw=1.6, CdWithCl=False)
    plot_polar_df(axes , df2 , marker=''  , label='CFD n=4'   , c=fColrs(4) , ls=':'  , lw=1  , CdWithCl=False)
    plot_polar_df(axes , df3 , marker=''  , label='CFD n=24'  , c=fColrs(2) , ls='--' , lw=1.2, CdWithCl=False)
    plot_polar_df(axes , df4 , marker=''  , label='CFD n=121' , c=fColrs(1) , ls='-'  , lw=1.5, CdWithCl=False)

    axes[0].set_xlabel(r'$\alpha$ [deg]')
    axes[0].set_ylabel('Coefficient [-]')
    axes[1].set_xlabel(r'$C_d$ [-]')
    axes[1].legend(fontsize=10, loc='lower right',
        frameon=False,# Remove the box border
        borderpad=0.1,# Tiny padding inside
        labelspacing=0.2,# Tiny vertical space between items
        handlelength=1.0,# Shorter line handles
        handletextpad=0.2,# Less space between handle and t
                   )
    axes[0].grid(True, ls=':', alpha=0.5)
    axes[1].grid(True, ls=':', alpha=0.5)
    ymin, ymax = axes[0].get_ylim()
    axes[0].set_xlim([-5, 22]) # OLD
    axes[0].set_ylim([-0.5, 2.35]) # OLD
    axes[1].set_xlim([-0.01, 0.12]) # OLD
#     axes[0].set_xlim([-3, 22])
#     axes[0].set_ylim([-0.1, 2.10])
#     axes[1].set_xlim([0.0, 0.05])
    fig.suptitle(airfoil_name.upper() + ' - Re = {}M'.format(Re), fontsize=13)
    for ax in axes:
        ax.tick_params(direction='in', top=True, right=True, labelright=False, labeltop=False, which='both')

    fig._title =f'polar_{airfoil_name}_Re{Re}'

    return fig

airfoil_names = []
airfoil_names += ['S809']
# airfoil_names += ['S809_15']
airfoil_names +=['du00-w-212']
airfoil_names +=['nlf1-0416']

for airfoil_name in airfoil_names:
    # Data Frames have columns Alpha, Cl, Cd
    if airfoil_name =='du00-w-212':
        Re=3
        dfe = standardize_polar_df(weio.read('_results/_polars/du00-w-212_re03.00M_EXP.csv').toDataFrame())
        df1 = standardize_polar_df(weio.read('_results/_polars/du00-w-212_re03.00M_CFD2D.csv').toDataFrame())
        df2 = standardize_polar_df(weio.read('_results/_polars/du00-w-212_re03.00M_CFD3D_n4.csv').toDataFrame())
        df3 = standardize_polar_df(weio.read('_results/_polars/du00-w-212_re03.00M_CFD3D_n24.csv').toDataFrame())
        df4 = standardize_polar_df(weio.read('_results/_polars/du00-w-212_re03.00M_CFD3D_n121.csv').toDataFrame())

    elif airfoil_name =='nlf1-0416':
        Re=4
        dfe = standardize_polar_df(weio.read('_results/_polars/nlf1-0416_re04.00M_EXP.csv').toDataFrame())
        df1 = standardize_polar_df(weio.read('_results/_polars/nlf1-0416_re04.00M_CFD2D.csv').toDataFrame())
        df2 = standardize_polar_df(weio.read('_results/_polars/nlf1-0416_re04.00M_CFD3D_n4.csv').toDataFrame())
        df3 = standardize_polar_df(weio.read('_results/_polars/nlf1-0416_re04.00M_CFD3D_n24.csv').toDataFrame())
        df4 = standardize_polar_df(weio.read('_results/_polars/nlf1-0416_re04.00M_CFD3D_n121.csv').toDataFrame())
# 
    elif airfoil_name =='S809':
        # NOTE: WE USE CLEAN, and CDW:
        # - The S809 is notorious for laminar bypass, even in the "Clean" case, which is why the clean conditions are better instead of the grit.
        # - OSU Grit were overtripped, resulting in massice spearation and thickneded Boundary layers that RANS will fail to capture accurately.
        # - Cdw captures the full momentum loss (skin friction and pressure)
        #
        Re=0.75
#         dfe = standardize_polar_df(weio.read('_results/_polars/S809_re00.75M_Exp.csv').toDataFrame())
#         dfe = standardize_polar_df(weio.read('_results/_polars/S809_re00.75M_Exp_Grit.csv').toDataFrame())
        dfe = standardize_polar_df(weio.read('_results/_polars/S809_re00.75M_EXP_CDW.csv').toDataFrame())
#         dfe = standardize_polar_df(weio.read('_results/_polars/S809_re00.75M_Exp_Grit_CDW.csv').toDataFrame())
        df1 = standardize_polar_df(weio.read('_results/_polars/S809_re00.75M_CFD2D.csv').toDataFrame())
        df2 = standardize_polar_df(weio.read('_results/_polars/S809_re00.75M_CFD3D_n4.csv').toDataFrame())
        df3 = standardize_polar_df(weio.read('_results/_polars/S809_re00.75M_CFD3D_n24.csv').toDataFrame())
        df4 = standardize_polar_df(weio.read('_results/_polars/S809_re00.75M_CFD3D_n121.csv').toDataFrame())

#     elif airfoil_name =='S809_15':
#         airfoil_name='S809'
#         Re=1.5
#         dfe = standardize_polar_df(weio.read('_results/_polars/S809_re01.50M_Exp.csv').toDataFrame())
# #         dfe = standardize_polar_df(weio.read('_results/_polars/S809_re01.50M_Exp_Grit.csv').toDataFrame())
#         dfe = standardize_polar_df(weio.read('_results/_polars/S809_re01.50M_EXP_CDW.csv').toDataFrame())
# #         dfe = standardize_polar_df(weio.read('_results/_polars/S809_re01.50M_EXP_Grit_CDW.csv').toDataFrame())
#         df1 = standardize_polar_df(weio.read('_results/_polars/S809_re01.50M_CFD2D.csv').toDataFrame())
#         df2 = standardize_polar_df(weio.read('_results/_polars/S809_re01.50M_CFD3D_n4.csv').toDataFrame())
#         df3 = standardize_polar_df(weio.read('_results/_polars/S809_re01.50M_CFD3D_n24.csv').toDataFrame())
#         df4 = standardize_polar_df(weio.read('_results/_polars/S809_re01.50M_CFD3D_n121.csv').toDataFrame())



    fig = plot_polars(airfoil_name, Re)

    axes=fig.axes


if export:
    export2pdf(twoByTwo=False)
# 

plt.show()



