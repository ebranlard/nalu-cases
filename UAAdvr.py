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
from welib.weio.fast_input_file import FASTInputFile, ADPolarFile
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
        self._dat = None  # See dat.setter
        self._dvr_filename = None 
        #self.dvr_filename = os.path.os.path.basename(dvr)
        self.sim_dir = sim_dir if sim_dir is not None else tempfile.gettempdir() 
        self._aeroTS = None
        self._inflowTS = None
        self._motionTS = None
        self._inflowTSFile = None
        self._motionTSFile = None
        self._dat_filename = None
        self._dat_modifs={}
        self._files_written=[]

        # ---
        self.set_template(dvr_template, dat)
        self.new() # Copy template and initialize simulation fields to zero

    def set_template(self, dvr0, dat0=None):
        self._dvr0 = FASTInputFile(dvr0)
        if dat0 is None:
            basedir = os.path.dirname(dvr0)
            dat0 = os.path.join(basedir, self._dvr0['AirFoil'].replace('"','')).replace('\\','/')
        self._dat0 = ADPolarFile(dat0)
        self._dvr0_filename = dvr0 # template filename cannot be overwritten
        self._dat0_filename = dat0 # template filename cannot be overwritten

    def new(self):
        # --- Copy template to local data"""
        self.dvr = self._dvr0.copy()
        self.dat = self._dat0.copy()
        # --- Initilize sim fields 
        self._aeroTS     = None
        self._inflowTS   = None
        self._motionTS   = None
        self._dat_modifs = {}
        self._files_written=[]

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
        #if key =='Airfoil':
        #    self._dat_filename = item
        #    self.dvr.__setitem__(key, item)

        #elif key =='MotionTSFile':
        #    self._motionTSFile = item
        #    self.dvr.__setitem__(key, item)

        if key in self.dvr:
            self.dvr.__setitem__(key, item)
        else:
            raise NotImplementedError()

    def __getitem__(self, key):
        if key in self.dvr:
            return self.dvr.__getitem__(key)
        else:
            raise NotImplementedError()

    # --------------------------------------------------------------------------------}
    # --- Properties Filenames
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
        if self._dat_filename is None:
            if self.dvr_filename is not None:
                return self.dvr_filename.replace('.dvr','_polar.dat') # TODO
        else:
            return self._dat_filename

    @dat_filename.setter 
    def dat_filename(self, value):
         self._dat_filename = value

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
        if self._motionTSFile is None:
            if self.dvr_filename is not None:
                return self.dvr_filename.replace('.dvr','_motionTS.csv')
        else:
            return self._motionTSFile
        
    @motionTSFile.setter 
    def motionTSFile(self, value):
         self._motionTSFile = value


    # --------------------------------------------------------------------------------}
    # --- Properties DataFrames
    # --------------------------------------------------------------------------------{
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

    # --------------------------------------------------------------------------------}
    # --- Properties in alphabetical order
    # --------------------------------------------------------------------------------{
    @property 
    def dat(self):
        return self._dat

    @dat.setter
    def dat(self, data):
        self.set_polar(data)

    @property
    def re(self):
        c   = self['Chord'] # Chord length (m)
        rho = self['FldDens']   # Density of working fluid (kg/m^3)
        nu  = self['KinVisc']   # Kinematic viscosity of working fluid (m^2/s)
        simMod = self['SimMod']
        U = self.U_ref
        re = U * c / nu
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


    # --------------------------------------------------------------------------------}
    # --- POLAR 
    # --------------------------------------------------------------------------------{
    def set_polar(self, data, Re=None):
        if isinstance(data, str):
            INFO('UAAdvr, setting polar from string')
            self._dat = ADPolarFile(data)

        elif isinstance(data, pd.DataFrame):
            from welib.airfoils.Polar import Polar
            INFO('UAAdvr, setting polar from dataframe')
            M = df.values
            pol = Polar(alpha=M[:,0], cl=M[:,1], cd=M[:,2], cm=M[:,3], Re=Re) 
            dat = pol.toAeroDyn(ADfilename, comment=comment, Re=Re) # Returns ADPolarFile
            self._dat = dat

        elif isinstance(data, ADPolarFile):
            self._dat = data

        elif isinstance(data, FASTInputFile):
            INFO('UAAdvr, setting polar from FASTInputFile')
            import pdb; pdb.set_trace()

    def update_polar(self, k, v):
        self._dat_modifs[k] = v # in the future, we could apply them at the end
        self.dat[k] = v

    # --------------------------------------------------------------------------------}
    # ---  
    # --------------------------------------------------------------------------------{
    def update(self, config):
        for param_name, param_value in self.sim_config.items():
           self.dvr[param_name] = param_value
           #self.dat[param_name] = param_value

#        self.dat['NumCoords'] = self.airfoil_coord
#         self.new_dat = os.path.join(self.sim_dir, f'{filebase}.dat')
#         self.dat.write(self.new_dat)



    def write(self, verbose=False):
        if self.dvr_filename is None:
            raise Exception('Cannot write, dvr_filename not set')

        base_dir = os.path.dirname(self.dvr_filename)

        # --- Set auxiliary files iin dvr
        if self['SimMod'] == 1:
            pass

        elif self['SimMod'] == 2:
            if verbose:
                print('Writing', self.aeroTSFile)
            self._files_written.append(self.aeroTSFile)
            self.aeroTS.to_csv(self.aeroTSFile, index=False)
            self['AeroTSFile'] = '"'+os.path.relpath(self.aeroTSFile, base_dir)+'"'

        elif self['SimMod'] == 3:
            if self['InflowMod'] == 2:
                if verbose:
                    print('Writing', self.inflowTSFile)
                self._files_written.append(self.inflowTSFile)
                self.inflowTS.to_csv(self.inflowTSFile, index=False)
                self['InflowTSFile'] = '"'+os.path.relpath(self.inflowTSFile, base_dir)+'"'

            if self['MotionMod'] == 2:
                if verbose:
                    print('Writing', self.motionTSFile)
                self._files_written.append(self.motionTSFile)
                self.motionTS.to_csv(self.motionTSFile, index=False)
                self['MotionTSFile'] = '"'+os.path.relpath(self.motionTSFile, base_dir)+'"'

        # --- Other DVR triggers
        self.dvr['AirFoil'] = '"'+os.path.relpath(self.dat_filename, base_dir)+'"'

        # --- Set dat
        self.dat['NumCoords'] = 0
        for k, v in self._dat_modifs.items():
            self.dat[k] = v
        if np.all(np.isnan(self.dat['AFCoeff'][:,3])):
            WARN('UAAdvr: Cm is NaN in polar, replacing it with 0')
            self.dat['AFCoeff'][:,3] = 0

        if verbose:
            print('Writing', self.dat_filename)
            print('Writing', self.dvr_filename)
        self._files_written.append(self.dat_filename)
        self._files_written.append(self.dvr_filename)
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


    def delete_input_files(self, ignoreList=None):
        for f in self._files_written:
            if f!=self._dvr0_filename and f!=self._dat0_filename:
                doDelete=True
                if ignoreList is not None:
                    for ii in ignoreList:
                        if ii in f: 
                            doDelete=False
                            break
                if doDelete:            
                    os.remove(f)


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
        dth  = np.gradient(theta, t)
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
