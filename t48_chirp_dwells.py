
""" 
Plot the chirp signal, with an example of Cl with UA and CFD

NOTE: s_factor = (2 * U) / chord   

s = t * s_factor = t * 2 U /chord

"""
import matplotlib.pyplot as plt
import numpy as np

import welib.weio as weio
from welib.weio.fast_output_file import FASTOutputFile
from welib.essentials import *
from welib.tools.figure import setFigureTitle
from welib.tools.colors import *

from helper_functions import load_json_chirp, plot_chirp_full_time, load_ULS
from helper_functions import split_chirp, postpro_chirp_tf, postpro_cycles_tf
from helper_functions import *

# --------------------------------------------------------------------------------}
# --- Helpers 
# --------------------------------------------------------------------------------{
def postpro_cycles_loops(dw, info, fig=None, ax=None, offset=0, pol=None, title=''):
    """ """
    if ax is not None:
        ax = ax
        fig=ax.figure
    else:
        if fig is None:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.8,4.3))
            fig.subplots_adjust(left=0.20, right=0.99, top=0.94, bottom=0.15, hspace=0.20, wspace=0.20)
        else:
            ax = fig.gca()

    COLRS = python_colors()
    STY=[':','-.','--','-','-',':']

    ALPHA=np.array([-1.5, 1.5])
    Cla = info['Cl_alpha'] * np.pi/180
    #if pol is not None:
    #    ax.plot(pol['Alpha_[deg]'], (pol['Cl_[-]']-offset)/Cla*100, 'k--')
    ax.plot(ALPHA, ALPHA, ':', lw=1.0, c='#e0e0e0')

    COLRS = { 0.1: '#87CEEB', 0.3: '#00BFFF', 0.5: '#4169E1', 1.0: '#191970' }
    BLUES = color_scales(5, color='blue', reverse=False)
    REDS  = darken_colors(color_scales(5, color='red' , reverse=False), 1)
    COLRS1= ['k', BLUES[-1], BLUES[-2], BLUES[-3]]
    COLRS2= [darken_color(REDS[-1],0.5), REDS [-1], lighten_color(REDS [-2], 0.1), lighten_color(REDS [-3],0.1)]
    #COLRS =['k', '#2166AC', '#67A9CF', '#D1E5F0']





    for i,d in enumerate(dw):
        c = d['cycles'][-1]
        k = d['k']
        # Make periodic
        c['th'] = np.concatenate([c['th'], [c['th'][0]]])
        c['cl'] = np.concatenate([c['cl'], [c['cl'][0]]])
        x = np.degrees(-c['th'])
        y = c['cl']/Cla


        alpha1=0.1+(i)*0.05
        res = get_loop_plot_data(x, y, alpha1=alpha1, alpha2=-alpha1)
        if res['rotation']=='Clockwise':
            COLRS=COLRS1
        else:
            COLRS=COLRS2
        #res = get_loop_plot_data(x, y, alpha1=None)
        ax.plot                                 (x, y,  c=COLRS[i], ls='-', alpha=0.2)
        ax.plot(x[res['idx_up']],  y[res['idx_up']]  , c=COLRS[i], ls='-' , label=f"k={d['k']}")
        ax.plot(x[res['idx_down']],y[res['idx_down']], c=COLRS[i], ls='-', alpha=0.2)
        ax.plot(x[res['idx_down']],y[res['idx_down']], c=COLRS[i], ls='--', alpha=0.5)

        # Add the tail-less arrows
        for a in [res['arrow_up'], res['arrow_down']]:
            if a:
                ax.annotate('', xy=(a['x'] + a['dx']*0.01, a['y'] + a['dy']*0.01), xytext=(a['x'], a['y']), arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        from nalulib.curves import contour_is_clockwise
        cw = contour_is_clockwise(x, y)
        print(f"k={d['k']} - Clockwise {cw}")

    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$ (C_l-C_{l,0})/C_{l,\alpha} $ [-]")
    ax.set_title(title)
    ax._title = 'Hysteresis '+ title
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-1.1, 1.1])
    ax.grid(ls=':', c='#e0e0e0')
    ax.tick_params(direction='in', top=True, right=True, labelright=False, labeltop=False, which='both')
#     plt.show()
#     plt.figure(figsize=(6, 6))
    return fig

def get_loop_plot_data(x, y, alpha1=-0.5, alpha2=0.5):
    """
    Analyzes pitching cycle data for plotting.
    Returns:
        dict: Indices for up/downstroke, rotation string, and arrow coordinates.
    """
    # --Identify Strokes
    # Upstroke is where alpha is increasing
    dx = np.gradient(x)
    idx_up = dx >= 0
    idx_down = dx <= 0

    # Identify alpha values common to both up and downstroke to find the gap
    # We interpolate to a common grid to find where the loop is "fattest"
    common_x = np.linspace(np.min(x), np.max(x), 100)
    y_up_interp = np.interp(common_x, x[idx_up], y[idx_up])
    y_down_interp = np.interp(common_x, x[idx_down], y[idx_down])
    

    # 2. Determine Rotation (Signed Area)
    # Area < 0 is Clockwise (CW) for y=Cl, x=alpha
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    rotation = "Clockwise" if area < 0 else "Counter-Clockwise"

    # 3. Find Exact Data Points for Arrows at alpha = +/- 0.5
    def get_arrow_pt(target_x, mask):
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return None
        # Find index in the original array closest to target
        #best_idx = valid_indices[len(valid_indices) // 2] # The middle of the stroke
        best_idx = valid_indices[np.argmin(np.abs(x[valid_indices] - target_x))]
        return { 'x': x[best_idx], 'y': y[best_idx], 'dx': dx[best_idx], 'dy': np.gradient(y)[best_idx] }

    def get_arc_pt(mask, dist=1.3):
        indices = np.where(mask)[0]
        if len(indices) < 2: return None
        
        # Segment the stroke
        xs, ys = x[indices], y[indices]
        dxs, dys = dx[indices], np.gradient(y)[indices]
        
        # Calculate incremental distances between points: ds = sqrt(dx^2 + dy^2)
        ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        cumulative_s = np.concatenate(([0], np.cumsum(ds)))
        s_max = cumulative_s[-1]
        print('>>>s', np.max(cumulative_s))
        
        # Find the index where we have traveled half the total distance of this stroke
        mid_idx_in_stroke = np.argmin(np.abs(cumulative_s - (s_max-dist)))
        
        actual_idx = indices[mid_idx_in_stroke]
        return { 'x': x[actual_idx], 'y': y[actual_idx], 
            'dx': dx[actual_idx], 
            'dy': np.gradient(y)[actual_idx]
        }  


    arrow_up = get_arrow_pt(alpha2, idx_up)
    arrow_down = get_arrow_pt(alpha1, idx_down)

#     if alpha1 is None:
#         gap = np.abs(y_up_interp - y_down_interp)
#         alpha1 = common_x[np.argmax(gap)]
#         alpha2=-alpha1


#     arrow_up = get_arc_pt(idx_down)

#     arrow_down =get_arc_pt(idx_up)

    return {
        "idx_up": idx_up,
        "idx_down": idx_down,
        "rotation": rotation,
        "area": area,
        "arrow_up": arrow_up,
        "arrow_down": arrow_down
    }



# --------------------------------------------------------------------------------}
# ---  
# --------------------------------------------------------------------------------{

setFigurePath('../nalu_torque_2026/figs/')
setFigureFont(12)
setFigureTitle(1)
export=True
# export=False


cases=[]
cases+=[{'airfoil_name':'S809'       , 'n':24 , 're':0.8 , 'suffix':'_HRCAT', 'Cl_alpha':6.250000, 'alpha0':-1.00410  }] # NOTE: 0.8 or 0.75
# cases+=[{'airfoil_name':'du00-w-212' , 'n':22 , 're':3   , 'suffix':'_HRCAT', 'Cl_alpha':6.43284, 'alpha0':-2.35240}] # <<<< INCOMPLETE SO FAR
# cases+=[{'airfoil_name':'nlf1-0416'  , 'n':24 , 're':4   , 'suffix':'_HRCAT', 'Cl_alpha':6.56495, 'alpha0':-3.93070}]
# cases+=[{'airfoil_name':'ffa-w3-211' , 'n':24 , 're':10  , 'suffix':'_HRCAT', 'Cl_alpha':6.76063, 'alpha0':-2.78090}]


# --------------------------------------------------------------------------------}
# --- Script  
# --------------------------------------------------------------------------------{

figc = None
fig0 = None
fig2 = None
fig5 = None
figl = None

# --- ULS
uls_json = f'_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_HRCAT.json'
uls_csv  = f'_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_HRCAT_ULS.csv'
cfd_outb = f'_results/cases_chirp_n24/S809/S809_re00.8_mean00_A01_HRCAT_CFD.outb'
dfc_uls = FASTOutputFile(cfd_outb).toDataFrame()
info_uls, dfm_uls = load_json_chirp(uls_json, verbose = False, plot = False)
info_uls['Cl_alpha']=6.250000
dfl = load_ULS(uls_csv, dfc_uls)
# dfl['Alpha_[deg]'] = -dfl['th'] # <<<< TODO


# --- Loop on cases
for cs in cases:
    # --- JSON Info
    base = cs['airfoil_name'] + '_re{:04.1f}_mean00_A01'.format(cs['re']) + cs['suffix']
    afn = cs['airfoil_name']
    n=cs['n']
    print('>>>>>>>>>> BASE ', base)

    json_path = f'_results/cases_chirp_n{n}/{afn}/{base}.json'
    cfd_outb  = f'_results/cases_chirp_n{n}/{afn}/{base}_CFD.outb'
    ua0_outb  = f'_results/cases_chirp_n{n}/{afn}/{base}_UA0_OF.outb'
    uaq_outb  = f'_results/cases_chirp_n{n}/{afn}/{base}_UA0_OF.outb'
    ua2_outb  = f'_results/cases_chirp_n{n}/{afn}/{base}_UA2_OF.outb'
    ua5_outb  = f'_results/cases_chirp_n{n}/{afn}/{base}_UA5_OF.outb'

    # --- JSON Info
    info, dfm    = load_json_chirp(json_path, verbose = False, plot = False)
    info.update(cs)

    # --- Read postprocessed CFD (see t41_chirp_ua)
    dfc = FASTOutputFile(cfd_outb).toDataFrame()
    dfq = FASTOutputFile(uaq_outb).toDataFrame()
    df0 = FASTOutputFile(ua0_outb).toDataFrame()
    df2 = FASTOutputFile(ua2_outb).toDataFrame()
    df5 = FASTOutputFile(ua5_outb).toDataFrame()

    dfq['Cl_[-]']= dfq['Cl_qs_Q_[-]'] # Cl at quarter chord


    # --- Remove Cl0
    t_ref = (info['indices_phases'][0]-10)*info['dt'] # Time at end of transients
    print('>>> t_ref', t_ref)
    it = np.argmin(np.abs(dfc['Time_[s]'].values - t_ref))
    dfc_ref = dfc['Cl_[-]'].values[it]
    dfq_ref = dfq['Cl_[-]'].values[it]
    df0_ref = df0['Cl_[-]'].values[it]
    df2_ref = df2['Cl_[-]'].values[it]
    df5_ref = df5['Cl_[-]'].values[it]
    print('Ref values: ', dfc_ref, df0_ref, df2_ref, df5_ref)

    dfc['Cl_[-]'] -= dfc_ref
    dfq['Cl_[-]'] -= dfq_ref
    df0['Cl_[-]'] -= df0_ref
    df2['Cl_[-]'] -= df2_ref
    df5['Cl_[-]'] -= df5_ref

    # --- Slip chirp and dwells
    _, trc, stc, chc, dwc = split_chirp(info, dfm, dfc, plot=False)
    _, trq, stq, chq, dwq = split_chirp(info, dfm, dfq, plot=False)
    _, tr0, st0, ch0, dw0 = split_chirp(info, dfm, df0, plot=False)
    _, tr2, st2, ch2, dw2 = split_chirp(info, dfm, df2, plot=False)
    _, tr5, st5, ch5, dw5 = split_chirp(info, dfm, df5, plot=False)
    _, trl, stl, chl, dwl = split_chirp(info_uls, dfm_uls, dfl, plot=False)


    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(12.8,3.7))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.14, hspace=0.20, wspace=0.07)
    postpro_cycles_loops(dwc, info, pol=None, offset=dfc_ref, title='CFD', ax=axes[0])
    postpro_cycles_loops(dw0, info, pol=None, offset=df0_ref, title='UA0', ax=axes[1])
    postpro_cycles_loops(dw2, info, pol=None, offset=df2_ref, title='UA2', ax=axes[2])
    postpro_cycles_loops(dw5, info, pol=None, offset=df5_ref, title='UA5', ax=axes[3])
    for ax in axes[1:]:
        ax.set_ylabel('')
    fig._title = 'HysteresisAll'



#     fig = postpro_cycles_loops(dwq, info, pol=None, offset=dfq_ref, title='UA0 AC')
#     figc = postpro_cycles_loops(dwc, info, pol=None, offset=dfc_ref, fig=figc, title='CFD')
#     fig0 = postpro_cycles_loops(dw0, info, pol=None, offset=df0_ref, fig=fig0, title='UA0')
#     fig2 = postpro_cycles_loops(dw2, info, pol=None, offset=df2_ref, fig=fig2, title='UA2')
#     fig5 = postpro_cycles_loops(dw5, info, pol=None, offset=df5_ref, fig=fig5, title='UA5')
#     figl = postpro_cycles_loops(dwl, info_uls, pol=None, offset=0, fig=figl, title='ULS')

if export:
    export2pdf()

plt.show()
