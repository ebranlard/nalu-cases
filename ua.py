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

def get_ua_mod0_tf(k, Cl_alpha=2*np.pi):
    """
    Returns magnitude and phase (degrees) for UA Mod 0 transfer function.
    H(jk) = Cl_alpha * (1 + jk)
    """
    # Transfer Function H = Cl_alpha + i * (Cl_alpha * k)
    h_complex = Cl_alpha * (1 + 1j * k)
    
    magnitude = np.abs(h_complex)
    # Phase in degrees (positive indicates a lead)
    phase_deg = np.angle(h_complex, deg=True)
    
    return magnitude, phase_deg



def get_analytical_tf(freqs, U=1, Cl_alpha=1, A1=0.165, A2 = 0.335, b1= 0.0455, b2=0.3, chord=1):
    """ 
        Tu: time constant [s], defined as chord / (2 * U)
        NOTE: k_red =  omega/ Tu
    """
    omegas = 2 * np.pi * freqs
    s     = 1j * omegas
    Tu    = chord / (2*U)
    k =  (np.pi * freqs * chord) / U  # [-]
    
    # Total Circulatory TF
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

def get_ua_mod4_tf(freqs, U=1, Cl_alpha=2*np.pi, A1=0.165, A2=0.335, b1=0.0455, b2=0.3, chord=1):
    """
    Transfer function from alpha at c/4 to Cl (UA Mod 4).
    Includes kinematic 3/4-chord transformation, Wagner lag, and added mass.
    """
    omegas = 2 * np.pi * freqs
    s = 1j * omegas
    Tu = chord / (2 * U)
    k =  (np.pi * freqs * chord) / U  # [-]
    
    # Kinematic mapping from c/4 to 3/4 chord (alpha_34 / alpha_14)
    # alpha_34 = alpha_14 + (dot_alpha * c/2) / U = alpha_14 * (1 + s*Tu)
    H_kinematic = (1 + s * Tu)

    # Circulatory response (Wagner approximation)
    # From Eq 8, 9, 13 in reference
    H_circ = Cl_alpha * ((1 - A1 - A2) + (b1 * A1) / (s * Tu + b1) + (b2 * A2) / (s * Tu + b2))
    H_circ *=  H_kinematic 
    
    # 4. Non-circulatory / Added Mass term (Eq 16: pi * Tu * omega)
    # Here omega = s * alpha_14
    H_nc = np.pi * Tu * s
    
    # Total Transfer Function H = Cl / alpha_14
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


def get_ua_mod3_tf(freqs, U=1, Cl_alpha=2*np.pi, A1=0.165, A2=0.335, b1=0.0455, b2=0.3, chord=1, filtCutOff=0.5, k_ta=0.75):
    """
    Refined UA Mod 3 TF including the AeroDyn low-pass filter 
    and non-circulatory deficiency terms.


    # NOTE: 
    #   if NC lift (ie. KC%Cn_alpha_q_nc) is set to 0, then the circulatory part below works well

        tau_lp = 1.0 / (2 * np.pi * f_cutoff) # Convert to time constant: tau = 1 / (2 * pi * f_cutoff)
        H_lp = 1.0 / (1.0 + s * tau_lp)
        H_circ = Cl_alpha * ((1 - A1 - A2) + (b1 * A1) / (s * Tu + b1) + (b2 * A2) / (s * Tu + b2))
        H_circ *= H_lp
    """
    #A1, A2, b1, b2 = 0.3, 0.7, 0.14, 0.53 # Standard Leishman values

    omegas = 2 * np.pi * freqs
    s = 1j * omegas
    Tu = chord / (2 * U)
    k =  (np.pi * freqs * chord) / U  # [-]
    
    # 1. Low-pass filter (Eq 1.8 in manual)
    # dynamicFilterCutoffHz = max(1.0, U) * filtCutOff / (PI * chord)
    f_cutoff = max(1.0, U) * filtCutOff / (np.pi * chord)  
    #dt = 0.00833333
    #LowPassConst  =  np.exp(-2*np.pi*dt*f_cutoff) # from Eqn 1.8 [7]
    tau_lp = 1.0 / (2 * np.pi * f_cutoff) # Convert to time constant: tau = 1 / (2 * pi * f_cutoff)
    # Filter Transfer Function
    H_lp = 1.0 / (1.0 + s * tau_lp)
    
    # 2. Kinematic + Circulatory (Wagner)
    H_kinematic = (1 + s * Tu)
    H_circ = Cl_alpha * ((1 - A1 - A2) + (b1 * A1) / (s * Tu + b1) + (b2 * A2) / (s * Tu + b2))
#     H_circ *=  H_kinematic 
    H_circ *= H_lp
    
    # 3. Non-circulatory (Deficiency part)
    # Based on Eq 1.18 - acting as a high-freq bleed
    C_nalpha=Cl_alpha
    a_s = 340.29
    M = U / a_s
    beta_M  = np.sqrt(1-M**2) 
    k_alpha = 1 / ( (1 - M) + (C_nalpha/2) * M**2 * beta_M * (A1*b1 + A2*b2) )  # Eqn 1.11a
    k_q     = 1 / ( (1 - M) +  C_nalpha    * M**2 * beta_M * (A1*b1 + A2*b2) )  # Eqn 1.11b   
    T_I     = chord / a_s                                                    # Eqn 1.11c
    T_alpha  = T_I * k_alpha * k_ta                                             # Eqn 1.10a
    T_q      = T_I * k_q     * k_ta
    #T_alpha2 = 0.75 * chord / (2 * U) # Mach simplified to 0
    H_nc_a = (4 * T_alpha * s) / (1 + s * T_alpha)
    H_nc_q = (-1 * T_q * s) / (1 + s * T_q)   * s * chord/U # * H_lp
    H_nc = H_nc_a + H_nc_q
    # Not: q = c/U * alpha_dot

    #KC%Kprime_alpha  = KC%T_alpha, xd%Kprime_alpha_minus1(i,j), KC%Kalpha_f, Kalpha_f_minus1 )    ! Eqn 1.18b
    #KC%Kprime_q      = KC%T_q    , xd%Kprime_q_minus1(i,j)    ,  KC%Kq_f   , Kq_f_minus1     )    ! Eqn 1.19b 
    #KC%Cn_alpha_nc   =  4*T_alpha * ( KC%Kalpha_f - KC%Kprime_alpha ) / M                                             ! Eqn 1.18a
    #KC%Cn_q_nc       = -1*T_q     * ( KC%Kq_f - KC%Kprime_q ) / M                                                        ! Eqn 1.19a

    H_lp2 = 1.0 / (1.0 + s * tau_lp)
    H_nc *= (H_lp2)**2
    #H_nc *= (H_lp2)
    
    # Total Transfer Function
    #H = H_circ + H_nc * 60 # For 0.75 and fc=5
    H = H_circ + H_nc * 60 # For 0.75

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
