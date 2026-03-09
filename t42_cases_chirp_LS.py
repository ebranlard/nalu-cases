""" 
Run Unsteady Lifting Surface (LS) simulations for the combined step / chirp / dwell cases.

It's easier to launch several simulations at different offset times, and then combine the results afterwards.

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 
from welib.essentials import *
import welib.weio as weio
from welib.vortilib.panelcodes.uLS import main

scriptDir = os.path.dirname(os.path.abspath(__file__))


def combine():
    from nalulib.nalu_input import NALUInputFile
    from welib.weio.csv_file import CSVFile
    def interpolate_to_time(df, time_ref):
        df_interp = pd.DataFrame({'Time': time_ref})
        for col in df.columns:
            if col != 'Time':
                df_interp[col] = np.interp(time_ref, df['Time'], df[col])
        return df_interp


    yml = NALUInputFile ('./_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_HR_no_motion.yaml')
    df1 = CSVFile('./_results/cases_chirp_n24/S809/LS/S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM_run1.csv').toDataFrame()
    df2 = CSVFile('./_results/cases_chirp_n24/S809/LS/S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM_run2.csv').toDataFrame()
    df3 = CSVFile('./_results/cases_chirp_n24/S809/LS/S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM_run3.csv').toDataFrame()
    df4 = CSVFile('./_results/cases_chirp_n24/S809/LS/S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM_run4.csv').toDataFrame()
    df5 = CSVFile('./_results/cases_chirp_n24/S809/LS/S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM_run5.csv').toDataFrame()

    time_ref = yml.time
    dfa = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    dfb = dfa.groupby('Time', as_index=False).mean()
    df  = interpolate_to_time(dfb, time_ref[1:])
    df.to_csv    ('./_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_HRCAT_ULS.csv', index=False)

    plt.figure(figsize=(10,6))
    plt.plot(df['Time'] ,  df['Cl'], label='Combined Interpolated', color='grey', linewidth=4)
    plt.plot(df1['Time'], df1['Cl'], '-', label='Sim 1', alpha=1.0, lw=1.5)
    plt.plot(df2['Time'], df2['Cl'], ':', label='Sim 2' , alpha=0.7, lw=3)
    plt.plot(df3['Time'], df3['Cl'], '-', label='Sim 3', alpha=0.7, lw=1.5)
    plt.plot(df4['Time'], df4['Cl'], '-', label='Sim 4', alpha=0.7, lw=1.5)
    plt.plot(df5['Time'], df5['Cl'], '-', label='Sim 5', alpha=0.7, lw=1.5)
    plt.xlabel('Time')
    plt.ylabel('CL')
    plt.legend()
    plt.grid(True)
    plt.show()


def run():
    #dfMotion = weio.read('C:/Work/CFD_airfoil/nalu-cases/_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01.json').toDataFrame()
    U0  = 6.0 # from JSON file
    rho = 1.2 # from JSON/ dvr
    #dfMotion = weio.read('C:/Work/CFD_airfoil/nalu-cases/_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_UAA_motionTS.csv').toDataFrame()
    dfMotion = weio.read('C:/Work/CFD_airfoil/nalu-cases/_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_HRCAT_UAA_motionTS.csv').toDataFrame()
    outputDir = '_results/cases_chirp_n24/S809/'
    print(dfMotion.keys())
    dfMotion = dfMotion.iloc[::5]
    alpha = 1.1931177300987128 # to give a Cl of approx 0.13084
    alpha = 0.0 
    nt = len(dfMotion)
    print(nt)


#     simName = 'S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM'
#     with Timer():
#         main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=-1)

#     simName = 'S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM_run2'
#     toffset=80
#     dfMotion['Time_[s]'] -=toffset
#     with Timer():
#         main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=-1, toffset=toffset)


#     simName = 'S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM_run3'
#     toffset=120
#     dfMotion['Time_[s]'] -=toffset
#     with Timer():
#         main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=-1, toffset=toffset)

#     simName = 'S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM_run4'
#     toffset=160
#     dfMotion['Time_[s]'] -=toffset
#     with Timer():
#         main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=-1, toffset=toffset)


    simName = 'S809_re00.8_mean00_A01_HRCAT_ULS_OmegaM_run5'
    toffset=200
    dfMotion['Time_[s]'] -=toffset
    with Timer():
        main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=-1, toffset=toffset)

# 
#     simName = 'S809_re00.8_mean00_A01_HRCAT_ULS_Omega0'
#     with Timer():
#         main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=0)

#     simName = 'S809_re00.8_mean00_A01_HRCAT_ULS_OmegaP'
#     with Timer():
#         main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=1)



if __name__ == "__main__":

#     run()


    combine()


