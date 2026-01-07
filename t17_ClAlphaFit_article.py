import numpy as np
import weio
import pandas as pd
from nalulib.nalu_forces import standardize_polar_df

def get_cl_alpha_slope(df, alpha_min=-2.5, alpha_max=10.5):
    """
    Computes the lift curve slope dCl/dAlpha using linear regression.
    """
    # Filter data for the linear region
    mask = (df['Alpha'] >= alpha_min) & (df['Alpha'] <= alpha_max)
    alpha_sub = df['Alpha'][mask].values
    cl_sub = df['Cl'][mask].values
    
    if len(alpha_sub) < 2:
        return np.nan
    
    # Perform linear fit: Cl = slope * Alpha + intercept
    slope, intercept = np.polyfit(alpha_sub, cl_sub, 1)
    return slope

def report_slopes(airfoil_name, data_dicts):
    """
    Calculates slopes and prints a comparison report.
    data_dicts: list of {'label': str, 'df': DataFrame}
    """
    results = {}
    
    # Calculate slopes
    for item in data_dicts:
        results[item['label']] = get_cl_alpha_slope(item['df'])
    
    exp_slope = results.get('Exp')
    
    print(f"\n--- Lift Curve Slope Report: {airfoil_name} ---")
    print(f"{'Case':<15} | {'Slope [1/deg]':<15} | {'Rel. Error [%]':<10}")
    print("-" * 45)
    
    for label, slope in results.items():
        if np.isnan(slope):
            error_str = "N/A"
        elif label == 'Exp':
            error_str = "0.00 (Ref)"
        else:
            rel_error = (slope - exp_slope) / exp_slope * 100
            error_str = f"{rel_error:+.2f}"
            
        print(f"{label:<15} | {slope:<15.5f} | {error_str:<10}")

# --- Main Execution Loop ---
airfoil_configs = [
    {'name': 'S809',       're': '00.80M', 're_label': 0.75},
    {'name': 'du00-w-212', 're': '03.00M', 're_label': 3},
    {'name': 'nlf1-0416',  're': '04.00M', 're_label': 4}
]

for cfg in airfoil_configs:
    name = cfg['name']
    re_str = cfg['re']
    
    # Load and standardize
    # Note: Using your naming convention from the provided code
    try:
        dfe = standardize_polar_df(weio.read(f'_results/_polars/{name}_re{re_str}_EXP.csv').toDataFrame())
    except:
        dfe = standardize_polar_df(weio.read(f'_results/_polars/{name}_re{re_str}_Exp_Clean.csv').toDataFrame())
    df1 = standardize_polar_df(weio.read(f'_results/_polars/{name}_re{re_str}_CFD2D.csv').toDataFrame())
    df2 = standardize_polar_df(weio.read(f'_results/_polars/{name}_re{re_str}_CFD3D_n4.csv').toDataFrame())
    df3 = standardize_polar_df(weio.read(f'_results/_polars/{name}_re{re_str}_CFD3D_n24.csv').toDataFrame())
    df4 = standardize_polar_df(weio.read(f'_results/_polars/{name}_re{re_str}_CFD3D_n121.csv').toDataFrame())
    
    data_to_compare = [
        {'label': 'Exp',        'df': dfe},
        {'label': 'CFD 2D',     'df': df1},
        {'label': 'CFD 3D n4',  'df': df2},
        {'label': 'CFD 3D n24', 'df': df3},
        {'label': 'CFD 3D n121','df': df4}
    ]
    
    report_slopes(name, data_to_compare)
        
#     except FileNotFoundError as e:
#         print(f"Skipping {name}: File not found.")
