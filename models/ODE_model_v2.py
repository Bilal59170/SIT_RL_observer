"""
Simulating
    y'(t) = f(y(t), u(t)) with y(t) = (E(t), M(t), F(t), M_s(t))

Modified mosquito model:
    dE/dt   = beta_E * F * (1 - E/K) * M/(M + gamma * M_s) - (nu_E + delta_E) * E
    dF/dt   = nu * nu_E * E - delta_F * F
    dM/dt   = (1 - nu) * nu_E * E - delta_M * M
    dM_s/dt = u - delta_S * M_s

"""

import numpy as np
import matplotlib.pyplot as plt

from config import *
from collections import deque

def normalize(x, xmin, xmax):
    """Transform x from [xmin, xmax] range to [-1, 1] range."""
    return (x - xmin) / (xmax - xmin) * 2.0 - 1.0


class SimODE_v2:
    """Simulation class to build an observer with Reinforcement Learning. Uses different controls.
    Last version of the ODE_model classes (SimODE_old, SimODE_v1, SimODE_v2)

    3 observation modes ("observer_mode" attribute): 
    - observer_mode="direct" : action = observer. (action of size 4)
    - observer_mode="luenberger_full" : total observation  y = x (C = Id_4).
            L is 4x4 matrix (action of size 16, flattened).
    - observer_mode="luenberger_partial" -> partial observation y = E (C = [1,0,0,0]).
            L is 4x1 matrix (action of size 4).
    """
    def __init__(
        self,
        u_types : dict,
        dt, 
        tmax, 
        obs_multi_scaling = True,
        rwd_y1=0,
        rwd_y123=0, 
        rwd_y4=0, 
        factor_K = 1, 
        y0_without_u0=None,

        # Observations
        obs_y_hat=False,
        obs_u=False,
        obs_E_state=False,
        obs_EMF = False,
        obs_Ms=False, 
        obs_time=False,
        obs_time_mode="absolute",
        obs_error_obs=False,

        # Action
        action_only_E=False, # Output
        action_only_EMF=False,
        min_value_obs=0, # minimal value authorized for the observer 
        observer_mode="direct", # "direct", "luenberger_full" ou "luenberger_partial"

        # History
        y_period_history=None,
        length_y_history: int = 1
        ):



        #  biological model parameters 
        self.beta_E  = PARAMETERS_MODEL_ODE['beta_E']
        self.nu_E    = PARAMETERS_MODEL_ODE['nu_E']
        self.nu      = PARAMETERS_MODEL_ODE['nu']
        self.delta_E = PARAMETERS_MODEL_ODE['delta_E']
        self.delta_F = PARAMETERS_MODEL_ODE['delta_F']
        self.delta_M = PARAMETERS_MODEL_ODE['delta_M']
        self.K       = PARAMETERS_MODEL_ODE['K']
        self.delta_s = PARAMETERS_MODEL_ODE['delta_s']
        self.gamma_s = PARAMETERS_MODEL_ODE['gamma_s']
    
        # Simulation parameters
        self.dt    = dt
        self.tmax  = tmax
        self.rwd_y1 = rwd_y1
        self.rwd_y123 = rwd_y123
        self.rwd_y4   = rwd_y4
        self.factor_K = factor_K

        # Observations
        self.obs_y_hat=obs_y_hat
        self.obs_u=obs_u
        self.obs_E_state=obs_E_state
        self.obs_EMF=obs_EMF
        self.obs_Ms=obs_Ms
        self.obs_time=obs_time
        self.obs_time_mode=obs_time_mode
        self.obs_error_obs = obs_error_obs
        
        # State
        self.n_state = 4

        # Action / observer mode 
        self.action_only_E = action_only_E
        self.action_only_EMF = action_only_EMF
        self.observer_mode = observer_mode
        valid_modes = ("direct", "luenberger_full", "luenberger_partial")
        if self.observer_mode not in valid_modes:
            raise ValueError(f"observer_mode doit être parmi {valid_modes}, reçu {observer_mode!r}.")

        if self.observer_mode == "direct":
            # action = y_hat
            if self.action_only_E:
                self.n_action = 1 # Only E_hat to output
            elif self.action_only_EMF:
                self.n_action = 3
            else:
                self.n_action = 4
            self.n_xhat = self.n_action
        elif self.observer_mode == "luenberger_full":
            # Full observation y = x (C = Id_4) : action = L (4x4) flattened.
            self.n_action = 4 * self.n_state  # 16
            # Observer approximate the full state (E, M, F, Ms).
            self.n_xhat = self.n_state
        elif self.observer_mode == "luenberger_partial":
            # Partial observation y = E (C = [1,0,0,0]) : action = L (4x1) flattened.
            self.n_action = self.n_state  # 4
            self.n_xhat = self.n_state

        self.min_value_obs = min_value_obs
        self.last_L = None  # last gain matrix used (Luenberger modes)

        # Security bounds on y_hat (Luenberger modes only) to prevent overflow (Nans) 
        # noticed at the beginning of the training
        self.xhat_clip = np.array([10*self.K, 10 * self.K, 10 * self.K, 100 * self.K])

        # Linear part of the dynamics
        self.A_lin = np.array([
            [-(self.delta_E + self.nu_E), 0.0,                 0.0,           0.0],
            [(1 - self.nu) * self.nu_E,   -self.delta_M,       0.0,           0.0],
            [self.nu * self.nu_E,          0.0,               -self.delta_F,  0.0],
            [0.0,                          0.0,                0.0,          -self.delta_s],
        ])

        # Historics
        if y_period_history is not None and not (self.obs_E_state) : 
            raise ValueError("Historic not implemented for observations other than E.")
        self.y_period_history = y_period_history # same period for any state historic
        self.length_y_history = length_y_history
        self.next_y_time = np.inf
        # E
        self.length_E_hist = length_y_history
        self._E_history = None # history of E, initialised in reset()
        self.next_E_time = np.inf
        # u 
        self.length_u_history = length_y_history
        self._u_history = None # history of u, initialised in reset()
        self.next_u_time = np.inf
        # y_hat
        self.length_y_hat_history = length_y_history  # same length as y_history
        self.next_y_hat_time = np.inf
        self._y_hat_history = None # initilised in reset()
        # error error_obs_obs = y_hat - y (error of the observer with full observation)
        self.length_error_obs_history = length_y_history
        self._error_obs_history = None  # initialised in reset()
        self.next_error_obs_time = np.inf


        # Control parameters
        self.u_types = u_types
        self.ctrl_mode_resurgence = 0 # set of the control is 0 or not to allow resurgence
        # Set the control function
        self.u_fct, self.current_u_type = choose_u(u_types, instance_ODE_model=self)

        # Other attributes (initialized in reset)
        self.t = None
        self.y = None
        self.y_hat = None
        self.t_lst = None
        self.y_lst = None
        self.y_hat_lst = None


        #### Reset of the simulation ####
        self.reset(y0_without_u0)

    
    ####################################
    # reset / update / history management
    #####################################

    def reset(self, y0_without_u0=None):
        '''Reset the simulation. Called at each episode.'''
        self.t = 0.0

        # Take another control for the next episode
        self.ctrl_mode_resurgence = 0 # Put back the control to inactive in the case of control for resurgence 
        self.u_fct, self.current_u_type = choose_u(self.u_types, instance_ODE_model=self)
        
        # Initializes (E, M, F, Ms)
        if y0_without_u0 is not None:
            if len(y0_without_u0)!=4:
                raise ValueError(f"y0_without_u0 should be a numpy array of shape (,{self.n_state})")
            emfms = np.array(y0_without_u0, dtype=float)
        else:
            emfms = np.random.uniform(low=0.0, high= self.factor_K * self.K, size=(self.n_state))

        # Initializes u with u(E, M, F, Ms, t=0)
        self.y = np.array([*emfms, self.u_fct(t=0, EMFMs=emfms)]) # full state of the system
        
        # Initializes randomly the observer xhat.
        self.y_hat = np.random.uniform(low=0.0, high= self.factor_K * self.K, size=(self.n_xhat))
        self.last_L = None
        self.t_lst = [self.t]
        self.y_lst = [np.copy(self.y)]
        self.y_hat_lst = [np.copy(self.y_hat)]

        ## History management
        if self.y_period_history is not None:
            self.next_y_time = self.y_period_history # next time to store the history of y
        else:
            self.next_y_time = np.inf 
        # E
        e0 = self.y[0]
        self._E_history = deque([e0] * self.length_E_hist, maxlen=self.length_E_hist)
        self.next_E_time = self.next_y_time
        # u
        self._u_history = deque([self.y[4]] * self.length_u_history, maxlen=self.length_u_history)
        self.next_u_time = self.next_y_time

        # y_hat
        # ATTENTION : unlike E (scalar), y_hat is mutable we use np.copy to avoid dependencies in the historic
        self._y_hat_history = deque(
            [np.copy(self.y_hat) for _ in range(self.length_y_hat_history)],
            maxlen=self.length_y_hat_history
        )
        self.next_y_hat_time = self.next_y_time

        # error_obs = y_hat - y
        if self.obs_E_state:
            error_obs_0 = np.array([self.y_hat[0] - self.y[0]]) # error on E only
        else:
            error_obs_0 = self.y_hat - self.y[:self.n_xhat] # error on all observed components
        self._error_obs_history = deque(
            [np.copy(error_obs_0) for _ in range(self.length_error_obs_history)],
            maxlen=self.length_error_obs_history
        )
        self.next_error_obs_time = self.next_y_time

    def dynamics(self, y):
        """
        y : [E, M, F, M_s, u]   (5 components, u = y[4])
        returns dy/dt for [E, M, F, M_s] only  (4 components)
        """
        E, M, F, Ms, u = y
        u = abs(u)

        denom  = M + self.gamma_s * Ms
        mating = 0.0 if denom <= 1e-12 else M / denom

        dE_dt  = self.beta_E * F * (1 - E / self.K) * mating - (self.nu_E + self.delta_E) * E
        dM_dt  = (1 - self.nu) * self.nu_E * E - self.delta_M * M
        dF_dt  = self.nu       * self.nu_E * E - self.delta_F * F
        dMs_dt = u - self.delta_s * Ms

        return np.array([dE_dt, dM_dt, dF_dt, dMs_dt])

    ##################################################
    # Observateur de Luenberger (modes non observer_mode != "direct")
    ##################################################

    def _safe_mating_ratio(self, M, Ms):
        """Computes M/(M + gamma_s*Ms) with numerical safeguards. 
        Usefull when M and M_s are close to 0
        """
        denom = M + self.gamma_s * Ms
        eps = 1e-6 * max(self.K, 1.0)
        if abs(denom) <= eps:
            return 0.0
        ratio = M / denom
        if not np.isfinite(ratio):
            return 0.0
        return float(np.clip(ratio, 0.0, 1.0))

    def _action_to_L(self, action):
        """Turns the RL action into a gain matrix  L.

        - observer_mode == "luenberger_full"    : action size 16 -> L (4,4)
        - observer_mode == "luenberger_partial" : action size 4  -> L (4,1)
        """
        action = np.asarray(action, dtype=float).flatten()
        if self.observer_mode == "luenberger_full":
            if action.size != 4 * self.n_state:
                raise ValueError(
                    f"action de taille {action.size}, attendu {4 * self.n_state} "
                    "(matrice L 4x4 aplatie) en mode luenberger_full."
                )
            return action.reshape(self.n_state, self.n_state)
        elif self.observer_mode == "luenberger_partial":
            if action.size != self.n_state:
                raise ValueError(
                    f"action de taille {action.size}, attendu {self.n_state} "
                    "(matrice L 4x1 aplatie) en mode luenberger_partial."
                )
            return action.reshape(self.n_state, 1)
        else:
            raise ValueError(
                f"_action_to_L ne doit être appelé qu'en mode Luenberger (observer_mode={self.observer_mode!r})."
            )

    def _rhs_luenberger_full(self, y_hat, y_true, u, L):
        """rhs = "right hand side". 
        Corresponds to xhat_dot for the full observation case y = x (C = Id_4).

        xhat_dot = A xhat + g(y, u) + L (xhat - y)

        g : non linearities +  control
        """
        E, M, F, Ms = y_true[:4]
        mating = self._safe_mating_ratio(M, Ms)
        carrying_factor = np.clip(1 - E / self.K, -10.0, 10.0)
        F_clipped = np.clip(F, -10.0 * self.K, 10.0 * self.K)
        g = np.array([
            self.beta_E * F_clipped * carrying_factor * mating,
            0.0,
            0.0,
            u,
        ])
        # C = Id_4 donc C @ y_hat - y = y_hat - y_true
        correction = L @ (y_hat - y_true[:4])
        return self.A_lin @ y_hat + g + correction

    def _rhs_luenberger_partial(self, y_hat, y_true, u, L):
        """rhs = "right hand side". 
        xhat_dot for the cas of partial observations y = E (C = [1,0,0,0]).

        xhat_dot = A xhat + B(f(xhat)) + D u + L (C xhat - y)
        """
        E_hat, M_hat, F_hat, Ms_hat = y_hat
        mating = self._safe_mating_ratio(M_hat, Ms_hat)
        # (1 - E_hat/K) est clippé pour éviter un overflow
        carrying_factor = np.clip(1 - E_hat / self.K, -10.0, 10.0)
        F_hat_clipped = np.clip(F_hat, -10.0 * self.K, 10.0 * self.K)
        f_val = self.beta_E * F_hat_clipped * carrying_factor * mating
        Bf = np.array([f_val, 0.0, 0.0, 0.0])
        Du = np.array([0.0, 0.0, 0.0, u])

        E_true = y_true[0]
        error_scalar = E_hat - E_true
        correction = L.flatten() * error_scalar

        return self.A_lin @ y_hat + Bf + Du + correction

    def update_observations_aux(self, period, state, history, t, next_time):
        """Auxiliary function to update observations based on a given period.

        - y_period None  : instantaneous update.
        - y_period defini : sample every y_period days ;
        """

        if period is None:
            # instantaneous update (retrocompatible with the case withou historic)
            last_obs = np.copy(state)
            history.append(last_obs)
            return 
        
        eps = 1e-12
        while t + eps >= next_time:
            last_obs = np.copy(state)
            history.append(last_obs)
            next_time += period
        
        return next_time

    def update_observation_y_hat(self):
        next_time = self.update_observations_aux(
            period=self.y_period_history,
            state=self.y_hat, history=self._y_hat_history, t=self.t,
            next_time=self.next_y_hat_time
        )
        if next_time is not None:
            self.next_y_hat_time = next_time
 
    def update_observation_E(self):
        next_time = self.update_observations_aux(
            period=self.y_period_history,
            state=self.y[0], history=self._E_history, t=self.t,
            next_time=self.next_E_time
        )
        if next_time is not None:
            self.next_E_time = next_time

    def update_observation_u(self):
        next_time = self.update_observations_aux(
            period=self.y_period_history,
            state=self.y[4], history=self._u_history, t=self.t,
            next_time=self.next_u_time
        )
        if next_time is not None:
            self.next_u_time = next_time

    def update_observation_error_obs(self):
        """Update the historic of the error e = y_hat - y[:n_xhat]."""
        if self.obs_E_state:
            error_obs = np.array([self.y_hat[0] - self.y[0]]) # error on E only
        else:
            error_obs = self.y_hat - self.y[:self.n_xhat] # error on all observed components
        next_time = self.update_observations_aux(
            period=self.y_period_history,
            state=error_obs,
            history=self._error_obs_history,
            t=self.t,
            next_time=self.next_error_obs_time
        )
        if next_time is not None:
            self.next_error_obs_time = next_time

    def update_y(self):
        '''One Euler step of the system.'''

        # Computation of (E, M, F, Ms)(t+dt) with with explicit Euler
        new_emfs = self.y[:4] + self.dt * self.dynamics(self.y)

        # Computation of u(t+dt) = k(t+dt, y(t+dt))  
        new_u    = self.u_fct(self.t, new_emfs) 

        # Rebuild full state (E, M, F, Ms, u)(t+dt)
        self.y   = np.append(new_emfs, new_u)

        # Update historic of quantities that only depend on y
        self.update_observation_E() # E(t) 
        self.update_observation_u() # u(t)

    def update_y_hat(self, action, y_true_prev=None, u_prev=None):
        """Update the observer. Used in env.py, not in step().

        - observer_mode == "direct" : y_hat = action.
        - observer_mode in {"luenberger_full", "luenberger_partial"} : Update of y_hat with 
          explicit Euler with the Luenberger-like dynamics . Uses
          (y_true_prev, u_prev) : state and control at the previous time
        """
        if self.observer_mode == "direct":
            self.y_hat = action # Direct approximation of the state : y_hat = action
        else:
            if y_true_prev is None or u_prev is None:
                raise ValueError(
                    "y_true_prev and u_prev are needed for update_y_hat in Luenberger mode "
                    "(There are given by step())."
                )
            L = self._action_to_L(action)
            self.last_L = L

            if self.observer_mode == "luenberger_full":
                rhs = self._rhs_luenberger_full(self.y_hat, y_true_prev, u_prev, L)
            elif self.observer_mode == "luenberger_partial":
                rhs = self._rhs_luenberger_partial(self.y_hat, y_true_prev, u_prev, L)
            else:
                raise ValueError(f"observer_mode unknown : {self.observer_mode!r}")

            self.y_hat = self.y_hat + self.dt * rhs
            # Numerical security 
            self.y_hat = np.nan_to_num(
                self.y_hat, nan=0.0, posinf=self.xhat_clip, neginf=0
            )
            self.y_hat = np.clip(self.y_hat, 0, self.xhat_clip)

        # Update historic of quantities that only depend on y_hat
        self.update_observation_y_hat()

    def step(self, action, store=False):
        """Execute one step of the simulation.

        store: if True, append (t, y) to t_lst/y_lst for later plotting.
               Keep False during training to avoid memory growth on long
               episodes (100k+ steps per episode).
        """
        # State and control at time t (before the update), needed in Luenberger modes
        y_true_prev = np.copy(self.y[:4])
        u_prev = self.y[4]

        # increment time
        self.t += self.dt
        self.update_y() # y(t+dt)
        self.update_y_hat(action, y_true_prev=y_true_prev, u_prev=u_prev)  # y_hat updated

        # Update the error error_obs_obs = y_hat - y[:n_xhat] for the observation
        self.update_observation_error_obs() 
        # save for later plotting/metrics only when requested
        if store:
            self.t_lst.append(self.t)
            self.y_lst.append(np.copy(self.y))
            self.y_hat_lst.append(np.copy(self.y_hat))
            
    
    ##################
    # Observations 
    ##################

    def _get_time_obs(self):
        """
        (Nail Baloul's function in SimODEdiscreteLuis) : return temporal observation in obs_time_mode.

        "absolute"     : position dans l'episode [0, tmax]
                         -> risque open-loop si T fixe
        "season"       : position dans l'annee [0, 365j]
                         -> signal saisonnier, pas de position dans l'episode
        "week_in_year" : semaine dans l'annee [0, 52]
                         -> plus realiste (granularite hebdomadaire)
        """
        if self.obs_time_mode == "season":
            return normalize(self.t % 365.0, 0.0, 365.0)
        elif self.obs_time_mode == "week_in_year":
            return normalize((self.t // 7.0) % 52.0, 0.0, 52.0)
        elif self.obs_time_mode == "absolute":  # "absolute" — comportement original
            tmax = self.tmax if self.tmax is not None else 1000.0
            return normalize(self.t, 0.0, tmax)
        else:
            raise ValueError("obs_time_mode value is wrong. Should be \"absolute\", \"season\", or \"week_in_year\".")

    def _append_obs_aux(self, state, value, list_scales, signed=False):
        """Auxiliary function for normalisation functions."""
        if signed: # e.g pour l'erreur e = y_hat - y, qui peut être négative
            for scale in list_scales:
                state.append(normalize(np.clip(value, -scale, scale), -scale, scale))
        else: # Pour des compartiments E, M, F qui sont positifs
            for scale in list_scales:
                state.append(normalize(np.clip(value, 0, scale), 0, scale))

    def _append_EMF_scaled(self, state, value, signed=False):
        """Multi scale normalisation for RL for  E, M, F values."""
        list_scales = [10 * self.K, 5 * self.K, self.K, self.K // 10, self.K // 100, self.K // 1000]
        self._append_obs_aux(state, value, list_scales=list_scales, signed=signed)

    def _append_Ms_scaled(self, state, value, signed=False):
            """Multi scale normalisation for RL for  Ms values."""
            list_scales = [100 * self.K, 50 * self.K, 10 * self.K, 5* self.K, self.K, self.K // 10, self.K // 100, self.K // 1000]
            self._append_obs_aux(state, value, list_scales=list_scales, signed=signed)   

    def _append_u_scaled(self, state, value, signed=False):
        """Multi scale normalisation for RL for u values."""
        list_scales = [50 * self.K, 10 * self.K, 5 * self.K, self.K, self.K // 10, self.K // 100]
        self._append_obs_aux(state, value, list_scales=list_scales, signed=signed)

    def _append_y_hat_vector(self, state, y_hat):
        """Add the observer as observation in the RL. Take a snapshot of y_hat,
          adapted to its size (1, 3 ou 4)."""
        n = len(y_hat)
        self._append_EMF_scaled(state, y_hat[0])       # E_hat
        if n >= 3:
            self._append_EMF_scaled(state, y_hat[1])   # M_hat
            self._append_EMF_scaled(state, y_hat[2])   # F_hat
            if n >= 4:
                self._append_Ms_scaled(state, y_hat[3])  # Ms_hat

    def _append_e_scaled(self, state, value):
        """Add the observation error e = h(y_hat) - h(y) as observation in the RL.
        Can be negative. 
        value : y - y_hat, size 1 if obs_E_state est True, 4 otherwise (full observation).
        """
        n = len(value) # length of the error vector. 1 or 4
        if self.obs_E_state: # Only E is observed, so the error = E_hat - E
            assert n == 1, "Error vector should have 1 component for E observation."
            self._append_EMF_scaled(state, value[0], signed=True)       # E_hat - E,
        else: # Full observation of y_hat, so the error = y_hat - y
            assert n == 4, "Error vector should have 4 components for full observation."
            self._append_EMF_scaled(state, value[0], signed=True)       # E_hat - E,
            self._append_EMF_scaled(state, value[1], signed=True)   # M_hat - M
            self._append_EMF_scaled(state, value[2], signed=True)   # F_hat - F
            self._append_Ms_scaled(state, value[3], signed=True)    # Ms_hat - Ms

        


    def get_obs(self):
        """Get the observations of the systems. 
        (History) Historics of observations can be added. The order of how
        it is added makes no difference for SB3 RL algorithm."""
        # Large scalling to have precise knowledge of observers. 
        state = []

        y = self.y

        # Time t
        if self.obs_time:
            state.append(self._get_time_obs())

        # Observer y_hat
        if self.obs_y_hat:
            for y_hat in self._y_hat_history:
                self._append_y_hat_vector(state, y_hat)

        # e = y_hat - y
        if self.obs_error_obs:
            for e_snap in self._error_obs_history:
                self._append_e_scaled(state, e_snap)

        # E
        if self.obs_E_state:
            for e_state in self._E_history:
                self._append_EMF_scaled(state, value=e_state)

        # EMF (not used in experiments with historic)
        if self.obs_EMF:
            e_obs, m_obs, f_obs = y[0], y[1], y[2]
            self._append_EMF_scaled(state, value=e_obs) # E
            self._append_EMF_scaled(state, value=m_obs) # M
            self._append_EMF_scaled(state, value=f_obs) # F

        # Ms (not used in experiments with historic)
        if self.obs_Ms:
            # Same rescaling as M+Ms in the control code
            ms = y[3]
            self._append_Ms_scaled(state, value=ms)

        # u
        if self.obs_u:
            for u_snap in self._u_history:
                self._append_u_scaled(state, value=u_snap)

        return np.array(state)  # no normalisation or multi-scale strategy for now
    
    ##############
    # Reward
    ##############

    def reward(self):
        reward = 0
        reward_info = {}

        # penalize norm of first three observer errors

        if self.rwd_y1 > 0:
            if self.observer_mode == "direct" and not(self.action_only_E):
                raise ValueError("rwd_y1 needs only one action in E, i.e action_only_E=True")
            diff_state_obs = self.y[0] - self.y_hat[0]
            r = -self.rwd_y1 * np.linalg.norm(diff_state_obs) / self.K
            reward_info["rwd_y1"] = r
            reward += r

        if self.rwd_y123 > 0:
            if self.observer_mode == "direct" and self.n_action < 3:
                raise ValueError("rwd_y123 requires at least 3 observed components (action_only_EMF or full mode)")
            diff_state_obs = self.y[:3] - self.y_hat[:3]
            r = -self.rwd_y123 * np.linalg.norm(diff_state_obs) / self.K
            reward_info["rwd_y123"] = r
            reward += r

        if self.rwd_y4 > 0:
            if self.observer_mode == "direct" and self.n_action < 3:
                raise ValueError("rwd_y4 requires the full 4-component observer")
            r = -self.rwd_y4 * np.linalg.norm(self.y[3]-self.y_hat[3]) / self.K
            reward_info["rwd_y4"] = r
            reward += r

        return reward, reward_info

    ##################
    # Other functions
    ##################

    def norm_y(self):
        """Compute the norm of the state without the control u."""
        return np.sum(self.y[:-1]**2)**0.5
    
    def simulate(self, n_steps):
        """Simulates the dynamics of the system. Used in tests to see the chosen control acting."""
        for _ in range(n_steps):
            self.step(action=np.zeros(self.n_action), store=True)  
    
    def plot(self, title="Mosquito model dynamics", title_ctrl = "", eq_point = None, bool_show=True):
        '''Plot trajectories without control.'''
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        array_t = np.array(self.t_lst)
        array_Y = np.array(self.y_lst)   # shape (T, 5)

        # Colours
        cE = 'tab:blue'
        cM = 'tab:orange'
        cF = 'tab:green'
        cMs = 'tab:red'
        cu = 'tab:brown'

        # --- First subplot : E, M, F ---
        axes[0].plot(array_t, array_Y[:, 0]/self.K, label='E(t)/K', color=cE)
        axes[0].plot(array_t, array_Y[:, 1]/self.K, label='M(t)/K', color=cM)
        axes[0].plot(array_t, array_Y[:, 2]/self.K, label='F(t)/K', color=cF)
        axes[0].set_title('E/K, M/K, F/K')
        axes[0].set_xlabel('Temps (jours)')
        axes[0].set_ylabel('Population')
        axes[0].legend()

        # --- Second subplot : Ms ---
        axes[1].plot(array_t, array_Y[:, 3]/self.K, label='Ms(t)', color=cMs)
        axes[1].set_title('Ms/K')
        axes[1].set_xlabel('Temps (jours)')
        axes[1].set_ylabel('Population')
        axes[1].legend()

        # --- Third subplot : u ---
        axes[2].plot(array_t, array_Y[:, 4], label='u(t)', color=cu)
        axes[2].set_title("u" + title_ctrl)
        axes[2].set_xlabel('Temps (jours)')
        axes[2].set_ylabel('Population')
        axes[2].legend()

        # Add the equilibrium point 
        if not(eq_point is None):
            axes[0].axhline(eq_point[0], linestyle='--', color=cE, label='E*')
            axes[0].axhline(eq_point[1], linestyle='--', color=cM, label='M*')
            axes[0].axhline(eq_point[2], linestyle='--', color=cF, label='F*')
            axes[1].axhline(eq_point[3], linestyle='--', color=cMs, label='Ms*')
            axes[2].axhline(eq_point[4], linestyle='--', color=cu, label='u*')

        fig.suptitle(title)
        plt.tight_layout()

        if bool_show:
            plt.show()
            
        return fig, axes