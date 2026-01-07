'''This class performs a simple simulation after updating some fixed parameters. 
It runs the simulation in a specific folder, if one is provided. 
The class can also take as input a specific DvR and DAT file, 
typically corresponding to a particular airfoil. 
"'''
import numpy as np
import pandas as pd
import os
import subprocess
import tempfile

# import shutil

# import re
# from openfast_toolbox.airfoils import Polar
# import pandas as pd
# from scipy.interpolate import UnivariateSpline
# import pickle as pkl
from welib.essentials import *
from welib.weio.fast_input_file import FASTInputFile
from welib.weio.csv_file import CSVFile

class TemplatedUAAdvr:
    #def __init__(self, sim_config, folder, given_dvr, given_dat, exe, airfoil_coord=0, filebase=None, sim_dir=None):
    def __init__(self, dvr_template, exe=None, dat=None, sim_dir=None):
        """ 
         INPUTS:
          - dvr: path of dvr input file name (template, cannot be overwritten)
          - dat: path of dat input file name (template, cannot be overwritten)
                 optional, can be inferred from dvr
          - exe: path of ua_dvr exe
        """
        self.exe = exe
        # Template data
        self._dvr0_filename = None # template filename cannot be overwritten
        self._dat0_filename = None # template filename cannot be overwritten
        self._dvr0 = None 
        self._dat0 = None 
        # --- Current simulation data
        self.dvr = None 
        self.dat = None 
        self._dvr_filename = None 
        #self.dvr_filename = os.path.os.path.basename(dvr)
        self.sim_dir = sim_dir if sim_dir is not None else tempfile.gettempdir() 
        self._aeroTS = None
        self._inflowTS = None
        self._motionTS = None

        # ---
        self.set_template(dvr_template, dat)
        self.copy_template()

    def set_template(self, dvr0, dat0=None):
        self._dvr0 = FASTInputFile(dvr0)
        if dat0 is None:
            basedir = os.path.dirname(dvr0)
            dat0 = os.path.join(basedir, self._dvr0['AirFoil'].replace('"','')).replace('\\','/')
        self._dat0 = FASTInputFile(dat0)
        self._dvr0_filename = dvr0 # template filename cannot be overwritten
        self._dat0_filename = dat0 # template filename cannot be overwritten

    def new(self):
        self.copy_template()

    def copy_template(self):
        """ Copy template to local data"""
        self.dvr = self._dvr0.copy()
        self.dat = self._dat0.copy()
        self._aeroTS = None
        self._inflowTS = None
        self._motionTS = None

    def __repr__(self):
        s = '<{} object> with attributes:\n'.format(type(self).__name__)
        s +=' - exe            : {}\n'.format(self.exe)
        s += ' - _dvr0_filename: {}\n'.format(self._dvr0_filename)
        s += ' - _dat0_filename: {}\n'.format(self._dat0_filename)
        s += ' - sim_dir     :   {}\n'.format(self.sim_dir)
        s += ' - dvr_filename:   {}\n'.format(self.dvr_filename)
        s += ' - dat_filename:   {}\n'.format(self.dat_filename)
        #s += " - dat.filename:   {}\n".format(self.dat.filename)
        if self.simMod==2:
            s += ' - aeroTSFile:   {}\n'.format(self.aeroTSFile)
        elif self.simMod==3:
            if self['InflowMod']==2:
                s += ' - inflowTSFile:   {}\n'.format(self.inflowTSFile)
            if self['MotionMod']==2:
                s += ' - motionTSFile:   {}\n'.format(self.motionTSFile)
        s += ' - dvr_filename:   {}\n'.format(self.dvr_filename)
        s +=f" - simMod      :   {self.simMod}\n"
        s +=f" - dvr['Chord']:   {self.dvr['Chord']}\n"
        s +=f" * U_ref        :   {self.U_ref}\n"
        s +=f" * re           :   {self.re}\n"
        s +=f"Useful functions:\n"
        s +=f" - run(filename)\n"
        return s

    def __setitem__(self, key, item):
        if key in self.dvr:
            return self.dvr.__setitem__(key, item)
        else:
            raise NotImplementedError()

    def __getitem__(self, key):
        if key in self.dvr:
            return self.dvr.__getitem__(key)
        else:
            raise NotImplementedError()

    # --------------------------------------------------------------------------------}
    # --- Properties 
    # --------------------------------------------------------------------------------{
    @property
    def dvr_filename(self):
        return self._dvr_filename

    @dvr_filename.setter
    def dvr_filename(self, filename):
        if filename == self._dvr0_filename:
            raise Exception('Refusing to set dvr_filename as the same as the template file')
        # Sanitize
        base, ext = os.path.splitext(filename)
        if ext.lower()!='.dvr':
            raise Exception('Use .dvr for extension')
        self._dvr_filename = filename

    @property 
    def dat_filename(self):
        if self.dvr_filename is not None:
            return self.dvr_filename.replace('.dvr','_polar.dat') # TODO

    @property 
    def aeroTSFile(self):
        if self.dvr_filename is not None:
            return self.dvr_filename.replace('.dvr','_aeroTS.csv')

    @property 
    def inflowTSFile(self):
        if self.dvr_filename is not None:
            return self.dvr_filename.replace('.dvr','_inflowTS.csv')

    @property 
    def motionTSFile(self):
        if self.dvr_filename is not None:
            return self.dvr_filename.replace('.dvr','_motionTS.csv')

    @property 
    def aeroTS(self):
        if self._aeroTS is None:
            if self['SimMod'] != 2:
                FAIL('Cannot access aeroTS if simMod !=2')
                return None
            df = CSVFile(self['AeroTSFile']).toDataFrame()
            df.columns = emptyAeroTS().columns
            self._aeroTS = df
        if self['SimMod'] == 2 and self._aeroTS is None:
            FAIL('aeroTS was not set')
        return self._aeroTS

    @property 
    def inflowTS(self):
        if self._inflowTS is None:
            if self['SimMod'] != 3 and self['InflowMod'] != 2:
                FAIL('Cannot access inflowTS if SimMod !=3 and InflowMod !=2')
                return None
            df = CSVFile(self['InflowTSFile']).toDataFrame()
            df.columns = emptyInflowTS().columns
            self._inflowTS = df
        if self['SimMod'] == 3 and  self['InflowMod'] == 2 and self._inflowTS is None:
            FAIL('inflowTS was not set')
        return self._inflowTS

    @property 
    def motionTS(self):
        if self._motionTS is None:
            if self['SimMod'] != 3 and self['MotionMod'] != 2:
                FAIL('Cannot access motionTS if SimMod !=3 and MotionMod !=2')
                return None
            df = CSVFile(self['MotionTSFile']).toDataFrame()
            df.columns = emptyMotionTS().columns
            self._motionTS = df
        if self['SimMod'] == 3 and self['MotionMod'] ==2 and self._motionTS is None:
            FAIL('motionTS was not set')
        return self._motionTS


    @aeroTS.setter 
    def aeroTS(self, df):
        if self['SimMod'] != 2:
            raise Exception('Cannot set aeroTS if simMod !=2')
        if len(df.columns)!=4:
            raise Exception(f'Dataframe for aeroTS should have 4 columns, input has {len(df.columns)}: {df.columns}')
        df = df.copy()
        df.columns = emptyAeroTS().columns
        self._aeroTS = df

    @inflowTS.setter 
    def inflowTS(self, df):
        if self['SimMod'] != 3 and self['InflowMod'] !=2:
            raise Exception('Cannot set inflowTS if SimMod !=3, and InflowMod !=2')
        if len(df.columns)!=3:
            raise Exception(f'Dataframe for inflowTS should have 4 columns, input has {len(df.columns)}: {df.columns}')
        df = df.copy()
        df.columns = emptyInflowTS().columns
        self._inflowTS = df

    @motionTS.setter 
    def motionTS(self, df):
        if self['SimMod'] != 3 and self['MotionMod'] !=2:
            raise Exception('Cannot set inflowTS if SimMod !=3, and MotionMod !=2')
        if len(df.columns)!=10:
            raise Exception(f'Dataframe for motionTS should have 10 columns, input has {len(df.columns)}: {df.columns}')
        df = df.copy()
        df.columns = emptyMotionTS().columns
        self._motionTS = df

    @property
    def re(self):
        c   = self['Chord'] # Chord length (m)
        rho = self['FldDens']   # Density of working fluid (kg/m^3)
        nu  = self['KinVisc']   # Kinematic viscosity of working fluid (m^2/s)
        simMod = self['SimMod']
        U = self.U_ref
        re = rho * U * c / nu
        return re

    @property 
    def simMod(self):
        return self.dvr['SimMod']

    @property
    def U_ref(self):
        U = np.nan
        if self['SimMod']==1:
            U = self['InflowVel']
        elif self['SimMod']==2:
            try:
                df = self.aeroTS
                U = df['InflowVel_[m/s]'].values.mean()
            except:
                 FAIL('Cannot access aero TS')
        elif self['SimMod']==3:
            if self['InflowMod']==1:
                Uvec = np.array(self['Inflow']).astype(float)
                U = np.linalg.norm(Uvec)
            else:
                try:
                    df = self.inflowTS
                    U = (df['Ux_[m/s]']**2+df['Uy_[m/s]']**2).values.mean()
                except:
                    FAIL('Cannot access inflow TS')
        return U
# 
#     def update_polar(self, df, calc_unsteady=False):
#         for key in ("AFCoeff", "NumAlf", "alpha0", "alpha1", "alpha2",
#                     "C_nalpha", "Cn1", "Cn2", "Cd0", "Cm0"):
#             try:
#                 del self.dat[key]
#             except KeyError:
#                 pass
#         df_unique = df.drop_duplicates(subset="Alpha_[deg]", keep="first")
#         df_unique = df_unique.sort_values("Alpha_[deg]").reset_index(drop=True)
#         arr = np.column_stack((
#             df_unique["Alpha_[deg]"].values,
#             df_unique["Cl_[-]"].values,
#             df_unique["Cd_[-]"].values,
#             df_unique["Cm_[-]"].values
#         ))
#         self.dat["AFCoeff"] = arr
#         self.dat["NumAlf"]   = arr.shape[0]
#         pol = Polar(alpha=arr[:,0],
#                     cl=arr[:,1],
#                     cd=arr[:,2],
#                     cm=arr[:,3],
#                     Re=self.sim_config['Re'])
#         try:
#             alpha0, alpha1, alpha2, cn_slope, cn1, cn2, cd0, cm0 = pol.unsteadyParams()
#         except:
#             print('FAIL UNSTEADY POALR')
#             raise 
#             
#         self.dat["alpha0"]    = round(alpha0, 4)
#         self.dat["alpha1"]    = round(alpha1, 4)
#         self.dat["alpha2"]    = round(alpha2, 4)
#         self.dat["C_nalpha"]  = round(cn_slope, 4)
#         self.dat["Cn1"]       = round(cn1, 4)
#         self.dat["Cn2"]       = round(cn2, 4)
#         self.dat["Cd0"]       = round(cd0, 4)
#         self.dat["Cm0"]       = round(cm0, 4)
#    
#     def getInflowVel(self, config):
#         Re = config['Re']
#         InflowVel = Re *1000000* self.dvr['KinVisc'] / self.dvr['Chord']
#         self.dvr['InflowVel'] = InflowVel

    def update(self, config):
        for param_name, param_value in self.sim_config.items():
           self.dvr[param_name] = param_value
           #self.dat[param_name] = param_value

#        self.dat['NumCoords'] = self.airfoil_coord
#         self.new_dat = os.path.join(self.sim_dir, f'{filebase}.dat')
#         self.dat.write(self.new_dat)
#         self.dvr['AirFoil'] = os.path.basename(self.new_dat)



    def write(self, verbose=False):
        if self.dvr_filename is None:
            raise Exception('Cannot write, dvr_filename not set')

        base_dir = os.path.dirname(self.dvr_filename)

        # Set auxiliary files
        if self['SimMod'] == 1:
            pass

        elif self['SimMod'] == 2:
            if verbose:
                print('Writing', self.aeroTSFile)
            self.aeroTS.to_csv(self.aeroTSFile, index=False)
            self['AeroTSFile'] = self.aeroTSFile

        elif self['SimMod'] == 3:
            if self['InflowMod'] == 2:
                if verbose:
                    print('Writing', self.inflowTSFile)
                self.inflowTS.to_csv(self.inflowTSFile, index=False)
                self['InflowTSFile'] = self.inflowTSFile

            if self['MotionMod'] == 2:
                if verbose:
                    print('Writing', self.motionTSFile)
                self.motionTS.to_csv(self.motionTSFile, index=False)
                self['MotionTSFile'] = os.path.relpath(self.motionTSFile, base_dir)
        
        if verbose:
            print('Writing', self.dat_filename)
            print('Writing', self.dvr_filename)

        self.dat['NumCoords'] = 0
        self.dat.write(self.dat_filename)
        self.dvr.write(self.dvr_filename)

# 
    def run(self, dvr_filename=None, verbose=False):
        if dvr_filename is not None:
           self.dvr_filename = dvr_filename
        if self.dvr_filename is None:
            raise Exception('Cannot run dvr_filename not set')

        self.write(verbose=verbose)
        command = [self.exe, self.dvr_filename]
        if verbose:
            print(f"Running command: {command}")
        subprocess.run(command, check=True)



    # --------------------------------------------------------------------------------}
    # --- Sim Mod specific 
    # --------------------------------------------------------------------------------{
    def setPrescribedSimMod3(self, tmax=None, dt=None, inflowdf=None, motiondf=None, ):
        if motiondf is not None:
            time = motiondf['Time_[s]']
        elif inflowdfdf is not None:
            time = motiondf['Time_[s]']
        if tmax is None:
            tmax = time.max()-time.min()
        if dt is None:
            dt =  (time.max()-time.min())/(len(time)-1)
        self['TMax']      = tmax
        self['DT']        = dt
        self['SimMod']    = 3
        if motiondf is not None:
            self['MotionMod'] = 2
            self['ActiveDOF'] = [False, False, False]
            self.motionTS = motiondf


    def setPrescribedPitchMotion(self, t, theta, simMod=3, dt=None):
        """ 
        - theta [rad]
        """
        dth = np.gradient(theta, t)
        ddth = np.gradient(dth, t)
        self['SimMod'] = simMod
        if simMod==3:
            df = emptyMotionTS(t)
            df['th_[rad]']       = theta
            df['dth_[rad/s]']    = dth
            df['ddth_[rad/s^2]'] = ddth

            self.setPrescribedSimMod3(motiondf = df, dt=dt)
        else:
            raise NotImplementedError()



#     
#     def sweep(self, idx):
#     # Aggiorna tutti i parametri della simulazione nei file di input
#         for param_name, param_value in self.sim_config.items():
#             self.dvr[param_name] = param_value
#             self.dat[param_name] = param_value
#         # Aggiorna anche le coordinate se serve
#         self.dat['NumCoords'] = self.airfoil_coord
#         filebase =f'AeroDyn_{idx}.dat'
#         self.dvr['AirFoil'] = filebase
# 

def emptyAeroTS(t=None):
    if t is None:
        t = np.zeros(0)
    return pd.DataFrame(data=np.column_stack((t, np.zeros((len(t), 3)))), columns = ['Time_[s]', 'Angle-of-attack_[deg]', 'InflowVel_[m/s]', 'Pitch_rate_[rad/s]'])

def emptyInflowTS(t=None):
    if t is None:
        t = np.zeros(0)
    return pd.DataFrame(data=np.column_stack((t, np.zeros((len(t), 2)))), columns = ['Time_[s]', 'Ux_[m/s]', 'Uy_[m/s]'])

def emptyMotionTS(t=None):
    if t is None:
        t = np.zeros(0)
    return pd.DataFrame(data=np.column_stack((t, np.zeros((len(t), 9)))), columns = ['Time_[s]', 'x_[m]', 'y_[m]', 'th_[rad]', 'dx_[m/s]', 'dy_[m/s]', 'dth_[rad/s]', 'ddx_[m/s^2]', 'ddy_[m/s^2]', 'ddth_[rad/s^2]'])


# 

if __name__ == '__main__':
    import json
    from helper_functions import load_json_chirp
    json_path = './_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01.json'
    info, dfc    = load_json_chirp(json_path, verbose = False, plot = False)
    #df_chirp = pd.DataFrame(data=np.column_stack([t_chirp, np.degrees(theta_chirp)]), columns=['Time_[s]','angle_[deg]'])

    exe = './_templates/UADVR/UnsteadyAero_Driver_x64.exe'
    dvr = './_templates/UADVR/S809.dvr'
    dvr = TemplatedUAAdvr(dvr_template=dvr, exe=exe)


    dvr['Chord']  = info['chord']
    dvr['FldDens']  = info['density']
    dvr['KinVisc']  = info['viscosity']
    dvr['Inflow'] = [0, info['U']]
    dvr['Vec_AQ'] = [0, 0] # from "A" to aerodynamic center Q. If "A" is at mid chord values are likely (0, -0.25) (-)
    dvr['Vec_AT'] = [0, 0.5] # from "A" to three-quarter chord "T". If "A" is at mid chord values are likely (0, 0.25) (-)

    sc=4
    dvr['GFScalingL1'] = [sc, 0.0, 0.0]
    dvr['GFScalingL2'] = [0.0, sc, 0.0]
    dvr['GFScalingL3'] = [0.0, 0.0, sc]


    print(dfc.keys())
    dvr.setPrescribedPitchMotion(t=dfc['Time_[s]'], theta = dfc['angle_[deg]']*np.pi/180, dt=info['dt'])
#     dvr['SimMod'] = 2
#     print(dvr)
    dvr.run('./_templates/test/S809.dvr', verbose=True)

    print(dvr)
    if (np.abs(dvr.re)-info['re']*1e6)>100:
        raise Exception('Problem in Re')
    print(info)
