import os

from welib.essentials import *
import welib.weio as weio
from welib.vortilib.panelcodes.uLS import main


if __name__ == "__main__":

    scriptDir = os.path.dirname(os.path.abspath(__file__))

    #dfMotion = weio.read('C:/Work/CFD_airfoil/nalu-cases/_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01.json').toDataFrame()
    U0  = 6.0 # from JSON file
    rho = 1.2 # from JSON/ dvr
    dfMotion = weio.read('C:/Work/CFD_airfoil/nalu-cases/_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_UAA_motionTS.csv').toDataFrame()
    outputDir = '_results/cases_chirp_n24/S809/'
    print(dfMotion.keys())
    dfMotion = dfMotion.iloc[::5]
    alpha = 1.1931177300987128 # to give a Cl of approx 0.13084
    alpha = 0.0 
    nt = len(dfMotion)
    print(nt)

#     simName = 'S809_re00.8_mean00_A01_ULS_OmegaM'
#     with Timer():
#         main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=-1)
# 
#     simName = 'S809_re00.8_mean00_A01_ULS_Omega0'
#     with Timer():
#         main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=0)

    simName = 'S809_re00.8_mean00_A01_ULS_OmegaP'
    with Timer():
        main(nChord=5, nSpan=3, nStep=nt, chord=1, span=12.0, U0=U0, rho=rho, alpha=alpha, dfMotion=dfMotion, outputDir=outputDir, simName=simName, motionType='body', omega_fact=1)
