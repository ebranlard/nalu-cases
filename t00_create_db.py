import pandas as pd
from nalulib.tools.dataframe_database import DataFrameDatabase
from nalulib.weio.csv_file import CSVFile

db_stat = DataFrameDatabase(name='Misc')

airfoil_name = 'du00-w-212'
config={'airfoil':airfoil_name, 'Re':3, 'density':1.225, 'viscosity':1.392416666666667e-05}
df1=CSVFile('./du00-w-212/ref/cfd_re3M.csv').toDataFrame()
df2=CSVFile('./du00-w-212/ref/cfd_re3M_turb.csv').toDataFrame()
df3=CSVFile('./du00-w-212/ref/exp_re3M.csv').toDataFrame()
configs=[]
configs+=[config |{'Setup':'CFD2U_ref_lam'}]
configs+=[config |{'Setup':'CFD2U_ref_turb'}]
configs+=[config |{'Setup':'EXP'}]
db_stat.insert_multiple(dfs=[df1, df2, df3], configs=configs)

# --- 
airfoil_name = 'nlf1-0416'
config={'airfoil':airfoil_name, 'Re':4, 'density':1.225, 'viscosity':1.0443125000000002e-05}
config['specific_dissipation_rate'] = 460.34999999999997
config['turbulent_ke']              = 0.00392448375
db_stat.insert(config|{'Setup':'CFD2D_ref_lam'},  CSVFile('./nlf1-0416/ref/cfd2d_rans.csv').toDataFrame())
db_stat.insert(config|{'Setup':'CFD2D_ref_turb'}, CSVFile('./nlf1-0416/ref/cfd2d_rans_turb.csv').toDataFrame())
db_stat.insert(config|{'Setup':'Exp'},            CSVFile('./nlf1-0416/ref/exp_Re4M.csv').toDataFrame())

# --- 
airfoil_name = 'ffa-w3-211'
config={'airfoil':airfoil_name, 'Re':10}
db_stat.insert(config|{'Setup':'none'}, pd.DataFrame())


self=db_stat



print(db_stat)
# print(db_stat.copy())
# --- Save db
db_stat.save('./experiments/DB_misc_stat.pkl')

# --- Load Glasgow 
db_stat_gl = DataFrameDatabase('experiments/glasgow/DB_exp_static.pkl')

# --- concatenate
db = db_stat_gl.concat(db_stat, inplace=False)

db.save('./experiments/DB_all_stat.pkl')
# # 
# # print(db_stat)
print(db)
import pdb; pdb.set_trace()
