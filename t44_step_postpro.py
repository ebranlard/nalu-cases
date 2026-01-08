""" 

Wagner: Response to an instantaneous change in pitch/angle of attack (the "Pitch Step"). 
The lift starts at a value (circulatory part is $0.5$ of steady-state theoretically) and grows toward $1.0$.

Küssner: Response to entering a sharp-edged gust (the "Transverse Gust"). The lift starts at $0$ and grows toward $1.0$.


CFD spike at t=0 is the non-circulatory (added mass) lift

Model       A1     b1    A2    b2      Sum A 
Wagner    0.165  0.045 0.335  0.300    0.5  # Jones
Küssner   0.500  0.130 0.500  1.000    1.0
OpenFAST  0.3    0.14  0.7    0.52     1.0  # OpenFAST

"""
import numpy as np
import os
import matplotlib.pyplot as plt
from welib.essentials import *
from welib.tools.pandalib import pd_interp1
from welib.weio.csv_file import CSVFile
from welib.weio.fast_output_file import FASTOutputFile
from helper_functions import load_json_chirp
from helper_functions import analyse_step_with_tStart
from helper_functions import analyse_step, load_ULS


# --------------------------------------------------------------------------------}
# --- MAIN SCRIPTS
# --------------------------------------------------------------------------------{
# --- Main inputs
cases=[]
cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':''    }]
cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HR' }]
cases+=[{'airfoil_name':'du00-w-212' , 'n':4  , 're':3   , 'suffix':''    }]
cases+=[{'airfoil_name':'du00-w-212' , 'n':22 , 're':3   , 'suffix':''    }]
cases+=[{'airfoil_name':'nlf1-0416'  , 'n':24 , 're':4   , 'suffix':''    }]
cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':''    }]
cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':'_HR' }]
# 


# --------------------------------------------------------------------------------}
# ---  Helpers
# --------------------------------------------------------------------------------{
chord = 1
span  = 4

pWag = [0.165,  0.045, 0.335,  0.300 ] # Wagner / Jones   Pitch change
pKus = [0.500,  0.13 , 0.500,  1.000 ] # Kussner          Transverse Gust
pOF  = [0.3  ,  0.14 , 0.7  ,   0.53 ] # OpenFAST

s_th = np.linspace(0, 100, 500)
PhiKussner = 1 - pKus[0]*np.exp(-pKus[1] * s_th) - pKus[2]*np.exp(-pKus[3]*s_th) # Kussner~ with A1+A2=1
PhiWagner  = 1 - pWag[0]*np.exp(-pWag[1] * s_th) - pWag[2]*np.exp(-pWag[3]*s_th) # Waggner
PhiOF      = 1 - pOF [0]*np.exp(-pOF [1] * s_th) - pOF [2]*np.exp(-pOF [3]*s_th) # OpenFAST


# --------------------------------------------------------------------------------}
# ---  
# --------------------------------------------------------------------------------{
fig=None
for cs in cases:
# for cs in [cases[0]]:
    base = cs['airfoil_name'] + '_re{:05.2f}M'.format(cs['re']) + cs['suffix']
    print(f"------------------------- {base}------------- {cs['suffix']}")
    yml_path  = '_results/cases_chirp_n{}/{}/{}_re{:04.1f}_mean00_A01{:s}.yaml'.format(cs['n'], cs['airfoil_name'], cs['airfoil_name'], cs['re'], cs['suffix'])
    json_path = yml_path.replace('.yaml','.json')
    dvr_path  = yml_path.replace('.yaml', '_UAA.dvr')
    cfd_outb  = yml_path.replace('.yaml', '_CFD.outb')
    uls_outb  = yml_path.replace('.yaml', '_ULS_OmegaM.csv')
    # --- JSON Info
    info, dfm = load_json_chirp(json_path, verbose = False, plot = False)

    dfc = FASTOutputFile(cfd_outb).toDataFrame()
    dfl = load_ULS(uls_outb, dfc)

    p,fig, _ = analyse_step(dfc, info, plot=True, label='CFD', c=fColrs(1), fig=fig, doFit=False)
    p,fig, _ = analyse_step(dfl, info, plot=True, label='ULS', c=fColrs(2), fig=fig, doFit=False)


    COLRSLB={'Wg':fColrs(3), 'Ks':fColrs(6), 'OF':fColrs(4)}
    LS = ['-', '--', ':', '-.']
#     fig=None
    #for im, UAMod in enumerate([2, 3, 4, 5]):
    for im, UAMod in enumerate([4, 2, 3]):
#         for ip, (p,lab) in enumerate(zip([pWag, pKus, pOF] , ['Wg', 'Ks', 'OF'])):
#         for ip, (p,lab) in enumerate(zip([pWag, pKus, pOF] , ['Wg', 'Ks', 'OF'])):
        for ip, (p,lab) in enumerate(zip([pWag, pKus, pOF] , ['OF'])):

            A1, b1, A2, b2 = p
            print(f"{lab:15s}: A1={A1:6.3f}, b1={b1:6.3f} A2={A2:6.3f}, b2={b2:6.3f}")
        #for ip, (p,lab) in enumerate(zip([pOF] , ['OF'])):
            uaa_outb  = yml_path.replace('.yaml', '_UA{}_{}.outb'.format(UAMod, lab))

            # --- Read postprocessed CFD (see t41_chirp_ua)
            dfu = FASTOutputFile(uaa_outb).toDataFrame()
            p,fig,_ = analyse_step(dfu, info, plot=True, label=f'UA{UAMod} {lab}', c=COLRSLB[lab], ls=LS[im], fig=fig, doFit=False, useCl=True)


ax=fig.axes[0]
ax.plot(s_th, PhiWagner , '-', label='(Wagner Function) ' , lw=2, c=fColrs(1))
# ax.plot(s_th, PhiKussner, '-', label='(Kussner Function)' , lw=2, c=fColrs(2))
# ax.plot(s_th, PhiOF     , '-', label='(OpenFAST Function)', lw=2, c=fColrs(3))
ax.legend()

plt.show()






