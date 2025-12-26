import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal.windows import tukey
from scipy.optimize import curve_fit



def tf_from_step(t, x, A=1):
    """
    Computes the Transfer Function (Gain/Phase) from a Step Response.
    
    Parameters:
    - t: Time vector for the step phase
    - x_: response
    - A: The magnitude of the jump
    Returns:
    - freqs: frequency vector [Hz]
    - mag: Magnitude (Gain)
    - phase: Phase (degrees)
    """
    dt = (t[-1] - t[0]) / (len(t) - 1)
    t = t.copy() - t[0]
    
    # 1. Compute the Impulse Response (Derivative of Step Response)
    # We divide by the amplitude of the input step to normalize it
    impulse_response = np.gradient(x, t) / A
    
    # 2. Windowing (Optional but recommended)
    # Applying a half-Tukey window can help if the signal hasn't 
    # perfectly settled, preventing "ringing" in the FFT.
    window = tukey(len(impulse_response), alpha=0.1)
    impulse_filtered = impulse_response * window
    
    # 3. Frequency Response via FFT
    n_fft = len(t) * 2 # Zero-padding for smoother curves
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    H = np.fft.rfft(impulse_filtered, n=n_fft)


    
    # 4. Convert to Aerodynamic Units
    mag = np.abs(H)
    phase = np.angle(H, deg=True)
    
    return freqs, mag, phase


def tfestimate_stitched(x, y, fs, f_stitch=1.5, returnAll=False):
    """
    Estimates TF using two different window lengths and stitches them.
    - f_stitch: Frequency (Hz) where we switch from long to short windows.
                Good idea to choose something around 10% of maximum frequency
    """
    # --- 1. Low-Frequency Pass (High Resolution) ---
    # We want few segments (e.g., 2) to maximize nperseg and resolve low f.
    n_low = len(x) // 2 
    nperseg_low = 2**int(np.log2(n_low))
    f_l, H_l = tfestimate(x, y, fs, nperseg=nperseg_low)
    
    # --- 2. High-Frequency Pass (Better Averaging / Less Smearing) ---
    # We want many segments (e.g., 16) to smooth out numerical noise.
    n_high = len(x) // 16
    nperseg_high = 2**int(np.log2(n_high))
    f_h, H_h = tfestimate(x, y, fs, nperseg=nperseg_high)
    
    # --- 3. Stitching Logic ---
    # Find indices where frequencies are below/above the stitch point
    idx_l = f_l <= f_stitch
    idx_h = f_h > f_stitch
    # Combine frequency arrays and transfer functions
    f_stitched = np.concatenate([f_l[idx_l], f_h[idx_h]])
    H_stitched = np.concatenate([H_l[idx_l], H_h[idx_h]])
    # Sort to ensure monotonicity (important for plotting/interpolation)
    sort_idx = np.argsort(f_stitched)
    if returnAll:
        return f_stitched[sort_idx], H_stitched[sort_idx], f_l, H_l, f_h, H_h
    else:
        return f_stitched[sort_idx], H_stitched[sort_idx]


def tfestimate(x, y, fs, nperseg=None, n_pad_factor=1, n_segments=8, returnCoh=False, verbose=True):
    """
    Replicates MATLAB's tfestimate using Scipy, with optional zero-padding for resolution.
    INPUTS:
     - n_pad_factor: multiplier for nperseg to increase frequency density.
    Returns:
        f: Frequency array
        H: Complex transfer function estimate
        coherence: Degree of linear relationship (0 to 1)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if nperseg is None:
            # Heuristic: Aim for ~8 segments, then round down to nearest power of 2 for FFT efficiency.
            n_ideal = len(x) // n_segments
            nperseg = 2**int(np.log2(n_ideal))  # Next power of 2
            nperseg = max(nperseg, 256) # Ensure it's not smaller than 256
            if verbose:
                print(f"Auto-selected nperseg: {nperseg} (Total samples: {len(x)})")
    # Define nfft for padding. 
    # This interpolates the spectrum, giving you more points (delta_f = fs/nfft)
    nfft = nperseg * n_pad_factor
    if verbose:
        print(f"nperseg: {nperseg} | nfft (padded): {nfft} | Total samples: {len(x)}")

    # Calculate PSD and CSD using nfft for zero-padding
    f, Pxx = signal.welch(x,    fs=fs, nperseg=nperseg, nfft=nfft)
    f, Pxy = signal.csd  (x, y, fs=fs, nperseg=nperseg, nfft=nfft)
    # The Transfer Function H = Pxy / Pxx
    H = Pxy / Pxx

    if returnCoh:
        # Magnitude Squared Coherence
        _, Cxy = signal.coherence(x, y, fs=fs, nperseg=nperseg, nfft=nfft)
        return f, H, Cxy
    else:
        return f, H




def cycle_mag_phase_fit(t, x, freq_target, plot=False, sine=True, ref=None):
    """
    Fits a sine wave to the output to find exact Mag and Phase.
    Returns positive magnitude and phase within 0 and 2*pi
    """
    # omega is known from your input
    t0 = t[0]
    om = 2 * np.pi * freq_target
    t = t.copy()-t0
    
    if sine:
        # Model: A*sin(om*t + phi) + offset
        def model(t_loc, A, phi, offset):
            return A * np.sin(om * t_loc + phi) + offset
    else:
        # Model: A*cos(om*t + phi) + offset
        def model(t_loc, A, phi, offset):
            return A * np.cos(om * t_loc + phi) + offset
        
    # Initial guess: [Amplitude, Phase, Mean]
    p0 = [(np.max(x) - np.min(x))/2, 0.0, np.mean(x)]
    
    try:
        params, _ = curve_fit(model, t, x, p0=p0)
        if params[0]<0:
            params[0]=-params[0]
            params[1]=np.mod( params[1] + np.pi, 2*np.pi)
    except Exception as e:
        print(f"Fit failed: {e}")
        params=[np.nan, np.nan, np.nan]
    
    xm = model(t, *params)
    if plot:

        fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        ax.plot(t+t0, x,  label='signal')
        ax.plot(t+t0, xm, label='fit')
        if ref is not None:
            ax.plot(t+t0, ref, label='input')
        ax.set_xlabel('Time ')
        ax.set_ylabel('')
        ax.legend()

    return params[0], params[1], xm # Return Amplitude, Phase (rad)


def tf_from_cycle(t, u, y, f_target, plot=False, sine=True):
    t0 = t[0]
    t = t.copy()-t0

    # Analyze Input 
    A_u, phi_u, u_fit = cycle_mag_phase_fit(t, u, f_target, sine=sine)

    # Analyze Output
    A_y, phi_y, y_fit = cycle_mag_phase_fit(t, y, f_target, sine=sine)

    # Transfer Function Magnitude and Phase
    mag = A_y / A_u
    # Phase difference (wrapped to -pi to pi)
    phi_rad = (phi_y - phi_u + np.pi) % (2 * np.pi) - np.pi # [rad]
    phi_deg = np.degrees(phi_rad)

    if plot:
        om = 2 * np.pi * f_target

        fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        ax.plot(t, u    , '-'    , label='Input'       , c='k')
        ax.plot(t, u_fit, '--'   , label='Input (fit)' )
        ax.plot(t, y    , '-'    , label='Output'      , c='b')
        ax.plot(t, y_fit, '--'   , label='Output (fit)')
        ax.plot(t, A_u * np.cos(om * t + phi_rad),  ':'   , label='Input (phased)' )
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('')
        ax.set_title(f'mag = {mag},  phi = {phi_deg} [deg]')
        ax.legend()

    info={'A_u':A_u, 'A_y':A_y, 'phi_u':phi_u, 'phi_y':phi_y, 'f':f_target, 'mag':mag, 'phi':phi_rad}
#           'u_fit':u_fit, 'y_fit':y_fit}

    return mag, phi_rad, info



