import numpy as np


def compute_bl_response(t, alpha, U=1, Cl_alpha=1, A1=0.165, A2 = 0.335, B1= 0.0455, B2=0.3, MACH=0.0, chord=1):
    """
    Computes B-L response using discrete-time state updates.
    Includes Circulatory (X, Y) and Non-circulatory (Impulsive) terms.
    INPUTS:
      - alpha: [rad] 
    """
    BETA_SQ = 1 - MACH**2
    # Non-circulatory Time Constant (State 3-5 approximation)
    # TI = c / speed_of_sound. Simplified as T_I = c/V for low speed lag.
    TI = chord / 343.0 
    # TI = 0 # 


    dt = t[1] - t[0]
    ds = (2 * U / chord) * dt # Non-dimensional time step

    
    # State initialization
    x1, x2, x3 = 0.0, 0.0, 0.0
    cl_out = np.zeros_like(alpha)
    
    # Pre-calculate alpha derivatives for impulsive terms
    da = np.gradient(alpha, dt)
    
    for i in range(1, len(t)):
        d_alpha = alpha[i] - alpha[i-1]
        
        # States 1 & 2: Circulatory Deficiency (First order lags)
        x1 = x1 * np.exp(-B1 * BETA_SQ * ds) + A1 * d_alpha * np.exp(-B1 * BETA_SQ * ds / 2)
        x2 = x2 * np.exp(-B2 * BETA_SQ * ds) + A2 * d_alpha * np.exp(-B2 * BETA_SQ * ds / 2)

        #xdot[0] = -1/Tu * (b1 + c * U_dot/(2*U**2)) * x1 + b1 * A1 / Tu * alpha_34
        #xdot[1] = -1/Tu * (b2 + c * U_dot/(2*U**2)) * x2 + b2 * A2 / Tu * alpha_34

        
        # Effective Angle of Attack
        alpha_e = alpha[i] - x1 - x2
        cl_circ = Cl_alpha * alpha_e
        
        # State 3: Non-circulatory (Impulsive/Added Mass) - Simplified lag
        # Represents the decay of the initial pressure spike
        x3 = x3 * np.exp(-dt/(TI)) + (da[i] - da[i-1]) * np.exp(-dt/(2*TI))
        cl_nc = (4 * TI * U / chord) * (da[i] - x3)
        
        cl_out[i] = cl_circ + cl_nc
        
    return cl_out


def get_analytical_tf(freqs, U=1, Cl_alpha=1, A1=0.165, A2 = 0.335, B1= 0.0455, B2=0.3, MACH=0.0, chord=1):

    BETA_SQ = 1 - MACH**2
    TI = chord / 343.0 

    # Non-dimensional Laplace variable
    # s = j * k, where k is reduced frequency (omega*c / 2V)
    omega_red = (2 * np.pi * freqs) * (chord / (2 * U))
    s = 1j * omega_red
    
    # Rate-based deficiency terms
    # beta_sq is usually 1.0 for low speed
    term1 = (A1 * s) / (s + B1 * BETA_SQ)
    term2 = (A2 * s) / (s + B2 * BETA_SQ)
    
    # Total Circulatory TF
    #H_circ = C_L_ALPHA * (1 - (A1 * B1) / (s + B1) - (A2 * B2) / (s + B2)) # OLD , state driven by alpha
    H_circ = Cl_alpha * (1 - term1 - term2) # State driven by Delta alpha

    # Non-circulatory part (High frequency addition)
    # Proportional to i*omega (derivative)
    H_nc = (4 * TI * U / chord) * (1j * 2 * np.pi * freqs)
    
    return H_circ + H_nc, H_circ, H_nc


