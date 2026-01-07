import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import weio

from nalulib.nalu_forces import standardize_polar_df, plot_polar_df
from welib.essentials import *
from welib.tools.figure import setFigureTitle

setFigurePath('../nalu_torque_2026/figs/')
setFigureFont(12)
setFigureTitle(1)
export=False
export=True


def plot_polars(airfoil_name, Re):
    fig,axes = plt.subplots(1, 2, sharey=True, figsize=(4.8,3.0))
    fig.subplots_adjust(left=0.16, right=0.99, top=0.92, bottom=0.17, hspace=0.20, wspace=0.08)
    plot_polar_df(axes, dfe, marker='o', label='Exp'     , c='k', ls='' , ms=3)
    plot_polar_df(axes, df1, marker='' , label='CFD 2D'  , c='k', ls='-', lw=1.6)
    plot_polar_df(axes, df2, marker='' , label='CFD 3D n=4'        , c=fColrs(4), ls=':' , lw=1)
    plot_polar_df(axes, df3, marker='' , label='CFD n=24'   , c=fColrs(2), ls='--', lw=1.2)
    plot_polar_df(axes, df4, marker='' , label='CFD n=121'  , c=fColrs(1), ls='-' , lw=1.5)

    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('Coefficient [-]')
    axes[1].set_xlabel('Cd [-]')
    axes[1].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, ls=':', alpha=0.5)
    axes[1].grid(True, ls=':', alpha=0.5)
    ymin, ymax = axes[0].get_ylim()
    axes[0].set_xlim([-5, 25])
    axes[0].set_ylim([-0.5, 2.35])
    axes[1].set_xlim([-0.01, 0.15])
    fig.suptitle(airfoil_name + ' - Re = {}M'.format(Re))
    for ax in axes:
        ax.tick_params(direction='in', top=True, right=True, labelright=False, labeltop=False, which='both')

    fig._title =f'polar_{airfoil_name}_Re{Re}'

    return fig

airfoil_names = ['S809']
airfoil_names +=['du00-w-212', 'nlf1-0416']

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

    elif airfoil_name =='S809':
        Re=0.75
        dfe = standardize_polar_df(weio.read('_results/_polars/S809_re00.80M_Exp_Clean.csv').toDataFrame())
        df1 = standardize_polar_df(weio.read('_results/_polars/S809_re00.80M_CFD2D.csv').toDataFrame())
        df2 = standardize_polar_df(weio.read('_results/_polars/S809_re00.80M_CFD3D_n4.csv').toDataFrame())
        df3 = standardize_polar_df(weio.read('_results/_polars/S809_re00.80M_CFD3D_n24.csv').toDataFrame())
        df4 = standardize_polar_df(weio.read('_results/_polars/S809_re00.80M_CFD3D_n121.csv').toDataFrame())

    fig = plot_polars(airfoil_name, Re)
    axes=fig.axes


if export:
    export2pdf(twoByTwo=False)
# 

plt.show()



