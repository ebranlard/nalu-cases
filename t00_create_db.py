import numpy as np
import pandas as pd
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.weio.csv_file import CSVFile

# --------------------------------------------------------------------------------}
# --- AMR Wind Benchmark cases 
# --------------------------------------------------------------------------------{
def create_amrbench_db():
    db = DataFrameDatabase(name='ARMBench')

    # --- 
    # Values from AMRWind Benchmark
    airfoil_name = 'du00-w-212'
    config={'airfoil':airfoil_name, 'Re':3, 'density':1.225, 'viscosity':1.392416666666667e-05} # NOTE: viscosity is mu
    config['specific_dissipation_rate']= 114.54981120000002      # Values from AMR Wind Bencmark
    config['turbulent_ke']             = 0.0013020495206400003 
    db.insert(config|{'Setup':'CFD2U_ref_lam'} ,  CSVFile('./airfoil_sims/du00-w-212/ref/cfd_re3M.csv').toDataFrame() )
    db.insert(config|{'Setup':'CFD2U_ref_turb'},  CSVFile('./airfoil_sims/du00-w-212/ref/cfd_re3M_turb.csv').toDataFrame() )
    db.insert(config|{'Setup':'EXP_turb'}      ,  CSVFile('./airfoil_sims/du00-w-212/ref/du00-w-212_re03.00M_EXP_turb.csv').toDataFrame() )
    db.insert(config|{'Setup':'EXP_lam'}       ,  CSVFile('./airfoil_sims/du00-w-212/ref/du00-w-212_re03.00M_EXP_lam_trans.csv').toDataFrame() )

    # --- 
    # Values from AMRWind Benchmark
    airfoil_name = 'nlf1-0416'
    config={'airfoil':airfoil_name, 'Re':4, 'density':1.225, 'viscosity':1.0443125000000002e-05} # NOTE: viscosity is mu
    config['specific_dissipation_rate'] = 460.34999999999997  # Values from AMR Wind Bencmark
    config['turbulent_ke']              = 0.00392448375  
    db.insert(config|{'Setup':'CFD2D_ref_lam'},  CSVFile('./airfoil_sims/nlf1-0416/ref/nlf1-0416_re04.00M_CFD2D_ref_rans_lam_trans.csv').toDataFrame())
    db.insert(config|{'Setup':'CFD2D_ref_turb'}, CSVFile('./airfoil_sims/nlf1-0416/ref/nlf1-0416_re04.00M_CFD2D_ref_rans_turb.csv').toDataFrame())
    db.insert(config|{'Setup':'EXP_turb'},       CSVFile('./airfoil_sims/nlf1-0416/ref/nlf1-0416_re04.00M_EXP_turb.csv').toDataFrame())
    db.insert(config|{'Setup':'EXP_lam'},        CSVFile('./airfoil_sims/nlf1-0416/ref/nlf1-0416_re04.00M_EXP_lam_trans.csv').toDataFrame())
    print(db)
    # --- Save db
    db.save('./airfoils_data/DB_amrbench_stat.pkl')
    return db


# --------------------------------------------------------------------------------}
# --- MISC
# --------------------------------------------------------------------------------{
def create_misc_db():
    db = DataFrameDatabase(name='Misc')
    # --- 
    airfoil_name = 'ffa-w3-211'
    config={'airfoil':airfoil_name, 'Re':10}
    config['density']                  = 1.2                    # TODO DUMMY VALUES
    config['viscosity']                = 9.0e-06 # This is mu   # TODO DUMMY VALUES
    config['specific_dissipation_rate']= 114.54981120000002     # TODO DUMMY VALUES
    config['turbulent_ke']             = 0.0013020495206400003  # TODO DUMMY VALUES
    db.insert(config|{'Setup':'none'}, pd.DataFrame())

    print(db)
    # --- Save db
    db.save('./airfoils_data/DB_misc_stat.pkl')

    return db


def create_IEA_db():
    arf_re = { 'fb90'             : 10,
               'fb80'             : 10,
               'fb70'             : 10,
               'fb60'             : 10,
               'snl-ffa-w3-560fb' : 10,
               'snl-ffa-w3-480fb' : 8,
               'snl-ffa-w3-420fb' : 10,
               'ffa-w3-360'       : 13,
               #'ffa-w3-330blend'  : 17,
               'ffa-w3-301'       : 18,
               #'ffa-w3-270blend'  : 17,
               'ffa-w3-241'       : 16,
               'ffa-w3-211'       : 10}

    db = DataFrameDatabase(name='IEA')

    for airfoil_name, re in arf_re.items():
        print(airfoil_name, re)

        config={'airfoil':airfoil_name, 'Re':re, 'density':1.225, 'viscosity':1.81e-05, 'Setup':'CFD2D_ref_Ellipsys'} # NOTE: viscosity is mu
        df = CSVFile(f'./airfoils_data/IEA22/{airfoil_name}_polar.csv').toDataFrame()
        df.columns = ['Alpha_[deg]', 'Cl_[-]', 'Cd_[-]', 'Cm_[-]']
        df['Alpha_[deg]'] *=180/np.pi
        db.insert(config, df)

    print(db)
    # print(db_stat.copy())
    # --- Save db
    db.save('./airfoils_data/DB_iea_stat.pkl')


# --------------------------------------------------------------------------------}
# --- NACA  
# --------------------------------------------------------------------------------{
def create_NACA_db():
    import welib.weio as weio

    db = DataFrameDatabase(name='NACA0018')

    # --- NACA0018
    # Read NACA0018 polars
    pols = weio.read('airfoils_data/_naca/__NACA_0018_AllRe.dat')
    dfs = pols.toDataFrame()
    reynolds = pols.fixedfile.reynolds 

    airfoil_name = 'naca0018'
    config={'airfoil':airfoil_name, 'Re':np.nan, 'density':1.225, 'viscosity':1.392416666666667e-05} # NOTE: viscosity is mu
    config['specific_dissipation_rate'] = 114.54981120000002     # TODO DUMMY VALUES
    config['turbulent_ke']              = 0.0013020495206400003  # TODO DUMMY VALUES

    for re in reynolds:
        db.insert(config | {'Re':re, 'Setup':'Unknown'}, dfs['AFCoeff_Re{:.2f}'.format(re)])
    print(db)
    # --- Save db
    db.save('./airfoils_data/DB_naca_stat.pkl')
    return db






def concat():
    # --- Load Static DBs
    ab = DataFrameDatabase('airfoils_data/DB_amrbench_stat.pkl')
    ms = DataFrameDatabase('airfoils_data/DB_misc_stat.pkl')
    gl = DataFrameDatabase('airfoils_data/glasgow/DB_exp_static.pkl')
    ia = DataFrameDatabase('airfoils_data/DB_iea_stat.pkl')

    # --- concatenate
    db = DataFrameDatabase.concatenate([gl, ab, ms, ia])
#     db = db_stat_gl.concat(db_stat_ms, inplace=False)
#     db = db.concat(db_stat_ia, inplace=False)
    db.save('./airfoils_data/DB_all_stat.pkl')
    db.configs.to_csv('./airfoils_data/DB_all_stat_configs.csv', index=False)
    print(db)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    create_NACA_db()
    #create_IEA_db()
    #create_amrbench_db()
    #create_misc_db()
    #concat()

    
