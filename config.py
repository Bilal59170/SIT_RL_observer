import numpy as np

# Variables globales utilisées dans le papier de Agbo Bidi, Lichté, Hayat, Coron
beta_E = 8.0
gamma_s = 1.0
nu_E = 0.25
delta_E = 0.03
delta_F = 0.04
delta_M = 0.1
delta_s = 0.12
nu = 0.49
K = 50000

PARAMETERS_MODEL_ODE = {
    'beta_E': beta_E,
    'nu_E': nu_E,
    'nu': nu,
    'delta_E': delta_E,
    'delta_F': delta_F,
    'delta_M': delta_M,
    'K': K,
    'delta_s': delta_s,
    'gamma_s': gamma_s
}


# Linear control main constants
R0 = beta_E * nu * nu_E / (delta_F * (delta_E + nu_E))
k_E_min = delta_s * (1 - nu) * nu_E * (R0 - 1) / (gamma_s * delta_M)
k_F_min = delta_F * delta_s * (1 - nu) / (delta_M * gamma_s * nu) * (R0 - 1)
# print("Minimal k for stablisation with u=kE: ", k_E_min) # ~13.23
# print("Minimal k for stablisation with u=kF: ", k_F_min) # ~4.32
# print("Minimal k for stablisation with u=k(M+Ms): ", k_MpMs_min)


## Function to choose u at each episode of the RL training
# Auxiliary functions
def choose_k_E():
    return np.random.uniform(0, 2 * k_E_min)
    
def choose_k_F():
    return np.random.uniform(0, 2 * k_F_min)

def resurgence_lin_ctrl(t, EMFMs, threshold_inf, threshold_sup, instance_ODE_model, k, idx_state):
    """Switching Linear control : 
     - Switch to 0 when the number of eggs E is too low to (<threshold_inf) allows population resurgence
     - Switch to linear control when the number of egg is too large (> threshold_sup) """
    if instance_ODE_model.ctrl_mode_resurgence==0: # ctrl is 0 to let population grow
        if EMFMs[0] > threshold_sup :
            instance_ODE_model.ctrl_mode_resurgence = 1
        return 0
    elif instance_ODE_model.ctrl_mode_resurgence==1:
        if EMFMs[0] < threshold_inf:
            instance_ODE_model.ctrl_mode_resurgence = 0
        return k * EMFMs[idx_state]
    else :
        raise ValueError("ctrl_mode_resurgence should be either 0 or 1")

# Main function
def choose_u(u_types, instance_ODE_model=None):
    """
        Function to choose u for the training of the observer RL model. It is called at each 
        beginning of an episode.
            - u_types : dict that indicates a probability distribution over the possible controls,
              e.g dict("u=kE":0.4, "u=kF":0.6).
    """
    
    keys = list(u_types.keys())
    probs = list(u_types.values())
    ctrl_type = np.random.choice(keys, p=probs)

    match ctrl_type:
        case "no_ctrl":
            ctrl_fct = lambda t, EMFMs: 0.0
        case "u=kE":
            k_E = choose_k_E()
            ctrl_fct = lambda t, EMFMs: k_E*EMFMs[0]
        case "u=kF":
            k_F = choose_k_F()
            ctrl_fct = lambda t, EMFMs: k_F*EMFMs[2]
        case "res_lin_det":
            # control that allows resurgence but with fixed deterministic thresholds. Used to test 
            # the RL model on approximation of identity E -> E.  Always taken with feedback in F.

            # choice of the linear control & threshold of population for E
            k, idx_state = 3 * k_F_min / 2, 2 # u=kF stabilisant en 0
            threshold_inf = 0.2 * K
            threshold_sup = 0.85 * K

            # Final control function
            ctrl_fct = lambda t, EMFMs: resurgence_lin_ctrl(
                t, EMFMs, threshold_inf, threshold_sup, instance_ODE_model, k, idx_state)
            
        case "res_lin":
            # control that allows resurgence but with non-deterministic thresholds to alow diversity.

            # choice of the linear control
            if np.random.randint(0, 2):
                k, idx_state = np.random.uniform(k_F_min, 2 * k_F_min), 2 # u=kF stabilisant en 0
            else:
                k, idx_state = np.random.uniform(k_E_min, 2 * k_E_min), 0 # u=kE stabilisant en 0

            # threshold of population for E
            threshold_inf = np.random.uniform(0.1 * K, 0.3 * K)
            threshold_sup = np.random.uniform(0.7 * K, K)

            # Final control function
            ctrl_fct = lambda t, EMFMs: resurgence_lin_ctrl(
                t, EMFMs, threshold_inf, threshold_sup, instance_ODE_model, k, idx_state)

        case _:
            raise ValueError("Unvalid control type")
    return ctrl_fct, ctrl_type

# Tests
if False:
    print("Test k : ", )
    for _ in range(5):
        print(f"-  kE = {choose_k_E()}, kF = {choose_k_F()}")