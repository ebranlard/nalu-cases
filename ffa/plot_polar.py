import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from welib.weio.csv_file import CSVFile

# Parameters
folders = []
folders += ['unity_static_polar']
folders += ['unity_static_overset_polar/']
U0 = 75.0  # Set your mean wind speed here
z = 4.0    # Span
chord =1  # Chord length
N = 10  # Number of last time steps to average
rho = 1.225  # Air density (kg/m^3)

# Helper to extract AoA from filename
def extract_aoa(filename):
    match = re.search(r'_aoa([-\d\.]+)', filename)
    if match:
        aoa_str = match.group(1).rstrip('.')
        try:
            return float(aoa_str)
        except ValueError:
            return None
    return None

# Collect results
plt.figure()

for folder in folders:
    results = []
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            print('fname:', fname)
            # Extract AoA from filename
            aoa = extract_aoa(fname)
            if aoa is None:
                continue
            #df = pd.read_csv(os.path.join(folder, fname))
            df = CSVFile(os.path.join(folder, fname)).toDataFrame()
            #print(df)
            Fx_tot = df['Fpx'].tail(N).mean() + df['Fvx'].tail(N).mean()
            Fy_tot = df['Fpy'].tail(N).mean() + df['Fvy'].tail(N).mean()
            # Non-dimensional coefficients
            S = z * chord    # Chord assumed 1, adjust if needed
            q_inf = 0.5 * rho * U0**2 * S
            Cx = Fx_tot / q_inf
            Cy = Fy_tot / q_inf
            print(aoa, Cx, Cy)
            results.append({'aoa': aoa, 'Cl': Cy, 'Cd': Cx})

    # Sort by AoA
    results = sorted(results, key=lambda x: x['aoa'])
    aoas = [r['aoa'] for r in results]
    Cls = [r['Cl'] for r in results]
    Cds = [r['Cd'] for r in results]

    # Plot
    plt.plot(aoas, Cls, label='Cl')
    plt.plot(aoas, Cds, label='Cd')
plt.xlabel('Angle of Attack (deg)')
plt.ylabel('Coefficient')
plt.legend()
plt.title('Cl and Cd vs Angle of Attack')
plt.grid(True)
plt.show()
