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


def get_analytical_tf(freqs, U=1, Cl_alpha=1, A1=0.165, A2 = 0.335, b1= 0.0455, b2=0.3, chord=1):
    """ 
        Tu: time constant [s], defined as chord / (2 * U)

    NOTE: k_red =  omega/ Tu

    """

    omegas = 2 * np.pi * freqs
    s     = 1j * omegas
    Tu    = chord / (2*U)
    k =  (np.pi * freqs * chord) / U  # [-]
    print('Tu', Tu)

    # Non-dimensional Laplace variable
    # s = j * k, where k is reduced frequency (omega*c / 2V)
    omega_red = omegas * (chord / (2 * U))
    s_red = 1j * omega_red
    
    # Rate-based deficiency terms
    # beta_sq is usually 1.0 for low speed
    term1 = (A1 * s_red) / (s_red + b1)
    term2 = (A2 * s_red) / (s_red + b2)
    
    # Total Circulatory TF
    #H_circ = Cl_alpha * (1 - (A1 * b1) / (s_red + b1) - (A2 * b2) / (s_red + b2)) # OLD , state driven by alpha
    #H_circ = Cl_alpha * (1 - term1 - term2) # State driven by Delta alpha
    H_circ = Cl_alpha * ((1 - A1 - A2) + (b1 * A1) / (s * Tu + b1) + (b2 * A2) / (s * Tu + b2))

    # Non-circulatory part: accounts for the pitch rate (added mass / damping) term
    # Non-circulatory part (High frequency addition)
    # Proportional to i*omega (derivative)
    #H_nc = (4 * TI * U / chord) * (1j * omegas)
    H_nc = np.pi * Tu * s 

    H = H_circ + H_nc

    out=dict()
    out['f']        = freqs
    out['k']        = k
    out['f_hmin']   = freqs[np.argmin(np.abs(H))]
    out['k_hmin']   = k    [np.argmin(np.abs(H))]
    print('k_hmin', out['k_hmin'], np.sqrt(((Cl_alpha *A2 * b2**2)/np.pi)**(1/3)-b2**2), np.sqrt((Cl_alpha *A2 * b2)/np.pi) )
    out['H']        = H
    out['mag']      = np.abs  (H)
    out['phi']      = np.angle(H,deg = True)
    out['H_circ']   = H_circ
    out['mag_circ'] = np.abs  (H_circ)
    out['phi_circ'] = np.angle(H_circ,deg = True)
    out['H_nc']     = H_nc
    out['mag_nc']   = np.abs  (H_nc)
    out['phi_nc']   = np.angle(H_nc,deg = True)
    out['p']        = [A1, b1, A2, b2]
    out['Tu']       = Tu
    out['U']        = U
    out['chord']    = chord
    return out

#     
#     return H_circ + H_non_circ
