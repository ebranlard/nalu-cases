""" 
Combine CFD rerun
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from welib.weio.fast_output_file import FASTOutputFile
from welib.weio.csv_file import CSVFile
from nalulib.nalu_input import NALUInputFile

def interpolate_to_time(df, time_ref):
    df_interp = pd.DataFrame({'Time': time_ref})
    for col in df.columns:
        if col != 'Time':
            df_interp[col] = np.interp(time_ref, df['Time'], df[col])
    return df_interp

def combine(yml, df1, df2, df3=None, plot=True):
    time_ref = yml.time
    dfa = pd.concat([df1, df2, df3], ignore_index=True)
    dfb = dfa.groupby('Time', as_index=False).mean()
    df  = interpolate_to_time(dfb, time_ref[1:])

    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(df['Time'], df['Fpy'], label='Combined Interpolated', color='grey', linewidth=4)
        plt.plot(df1['Time'], df1['Fpy'], '-', label='Sim 1', alpha=1.0, lw=1.5)
        plt.plot(df2['Time'], df2['Fpy'], ':', label='Sim 2' , alpha=0.7, lw=3)
        plt.plot(df3['Time'], df3['Fpy'], '-', label='Sim 3', alpha=0.7, lw=1.5)


        plt.xlabel('Time')
        plt.ylabel('Fpy')
        plt.title('Fpy from CFD Simulations')
        plt.legend()
        plt.grid(True)
    return df


# 
yml = NALUInputFile ('./_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_HR_no_motion.yaml')
df1 = CSVFile('./_results/cases_chirp_n24/S809/forces_S809_re00.8_mean00_A01_HR.csv').toDataFrame()
df2 = CSVFile('./_results/cases_chirp_n24/S809/forces_S809_re00.8_mean00_A01_HR_2.csv').toDataFrame()
df3 = CSVFile('./_results/cases_chirp_n24/S809/forces_S809_re00.8_mean00_A01_HR_3.csv').toDataFrame()
df = combine(yml, df1, df2, df3, plot=True)
df.to_csv    ('./_results/cases_chirp_n24/S809/forces_S809_re00.8_mean00_A01_HRCAT.csv', index=False)
print(df.shape, yml.time.shape, yml.time_dict)

# 
yml = NALUInputFile ('./_results/cases_chirp_n24/ffa-w3-211/ffa-w3-211_re10.0_mean00_A01_HR_no_motion.yaml')
df1 = CSVFile       ('./_results/cases_chirp_n24/ffa-w3-211/forces_ffa-w3-211_re10.0_mean00_A01_HR.csv').toDataFrame()
df2 = CSVFile       ('./_results/cases_chirp_n24/ffa-w3-211/forces_ffa-w3-211_re10.0_mean00_A01_HR_2.csv').toDataFrame()
df3 = CSVFile       ('./_results/cases_chirp_n24/ffa-w3-211/forces_ffa-w3-211_re10.0_mean00_A01_HR_3.csv').toDataFrame()
df = combine(yml, df1, df2, df3, plot=True)
df.to_csv          ('./_results/cases_chirp_n24/ffa-w3-211/forces_ffa-w3-211_re10.0_mean00_A01_HRCAT.csv', index=False)
print(df.shape, yml.time.shape, yml.time_dict)




plt.show()
