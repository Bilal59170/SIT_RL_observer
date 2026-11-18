"""
Simulating
    y'(t) = f(y(t), u(t)) with y(t) = (E(t), M(t), F(t), M_s(t))

Modified mosquito model:
    dE/dt   = beta_E * F * (1 - E/K)  - (nu_E + delta_E) * E
    dF/dt   = nu * nu_E * E - delta_F * F
    dM/dt   = (1 - nu) * nu_E * E - delta_M * M

Main observation (Total number of eggs, not ovitrap collections): E

"""

import numpy as np
import matplotlib.pyplot as plt

from config import *

def normalize(x, xmin, xmax):
    """Transform x from [xmin, xmax] range to [-1, 1] range."""
    return (x - xmin) / (xmax - xmin) * 2.0 - 1.0


class SimODE_v1:
    """Classe simple pour faire des tests sans contrôle. Devenue obsolète car utiliser 
    une classe avec contrôle u=0 revient à la même chose."""
    def __init__(
        self,
        dt, 
        tmax, 
        rwd_y1=0,
        rwd_y123=0, 
        factor_K = 1, 
        y0=None,
        # Observations
        obs_time=False,
        obs_time_mode="absolute",
        obs_E=True,
        obs_multi_scaling = True, # Observation scaling
        action_only_E = False, # Output
        min_value_obs=0
        ):

        #  biological parameters of the model
        self.beta_E  = PARAMETERS_MODEL_ODE['beta_E']
        self.nu_E    = PARAMETERS_MODEL_ODE['nu_E']
        self.nu      = PARAMETERS_MODEL_ODE['nu']
        self.delta_E = PARAMETERS_MODEL_ODE['delta_E']
        self.delta_F = PARAMETERS_MODEL_ODE['delta_F']
        self.delta_M = PARAMETERS_MODEL_ODE['delta_M']
        self.K       = PARAMETERS_MODEL_ODE['K']
        
        # Simulation parameters
        self.dt    = dt
        self.tmax  = tmax
        self.rwd_y1 = rwd_y1
        if action_only_E and self.rwd_y1 <= 0:
            raise ValueError("rwd_y1 should be positive if action_only_E is True")
        self.rwd_y123 = rwd_y123
        self.factor_K = factor_K

        # Observations
        self.obs_time = obs_time
        self.obs_time_mode = obs_time_mode
        self.obs_E = obs_E
        self.obs_multi_scaling = obs_multi_scaling # Several scalings of the observation 

        # Shape of the state and of the action
        self.n_state = 3
        self.n_action = 3
        self.action_only_E = action_only_E
        if self.action_only_E:
            self.n_action = 1 # Only E_hat to output
        
        # Other
        self.min_value_obs = min_value_obs # minimum value of the observer. Not usefull here but defined for retrocompatibility
        #### Reset of the simulation ####
        self.reset(y0)

        
    def reset(self, y0=None):
        '''Reset the simulation. Called at each episode.'''
        self.t = 0.0

        # Initialize (E, M, F)
        if y0 is not None:
            if len(y0)!=self.n_state:
                raise ValueError(f"y0 should be a numpy array of shape (,{self.n_state})")
            self.y = np.array(y0, dtype=float)
        else:
            self.y = np.random.uniform(low=0.0, high= self.factor_K * self.K, size=(self.n_state))
        
        # Initialize randomly the observer (E, M, F)_hat  that is constructed with reinforcement learning
        self.y_hat = np.random.uniform(low=0.0, high= self.factor_K * self.K, size=(self.n_action)) 

        self.t_lst = [self.t]
        self.y_lst = [np.copy(self.y)]
        self.y_hat_lst = [np.copy(self.y_hat)]

    def dynamics(self, y):
        """
        y : [E, M, F]   
        returns dy/dt for [E, M, F] 
        """
        E, M, F= y

        dE_dt  = self.beta_E * F * (1 - E / self.K) - (self.nu_E + self.delta_E) * E
        dM_dt  = (1 - self.nu) * self.nu_E * E - self.delta_M * M
        dF_dt  = self.nu       * self.nu_E * E - self.delta_F * F

        return np.array([dE_dt, dM_dt, dF_dt])

    def update_y_and_obs(self, action):
        '''Update the state y with one Euler step, and all the considered observations.'''
        # Computation of y(t+dt) with explicit Euler
        self.y = self.y + self.dt * self.dynamics(self.y)

        #### Update the observer ####
        self.y_hat = action  # Direct approximation of the state : x_hat = action 

    def step(self, action, store=False):
        """Execute one step of the simulation. Similar to step method in Simulation class for the
         RL for control code (cf Nathan & Nail's code)

        store: if True, append (t, y) to t_lst/y_lst for later plotting.
               Keep False during training to avoid memory growth on long
               episodes (100k+ steps per episode).
        """
        # Increment time
        self.t += self.dt
        self.update_y_and_obs(action)

        # save for later plotting/metrics only when requested
        if store:
            self.t_lst.append(self.t)
            self.y_lst.append(np.copy(self.y))
            self.y_hat_lst.append(np.copy(self.y_hat))

    def simulate(self, n_steps):
        """Simulates the dynamics of the system. Used in tests to see the chosen control acting."""
        for _ in range(n_steps):
            # action = np.zeros(self.n_action) : fake action to only focus on real state
            self.step(action=np.zeros(self.n_action), store=True) 

    def _get_time_obs(self):
        """
        (Reprend la fonction de Nail dans SimODEdiscreteLuis) : Retourne l'observation 
        temporelle selon obs_time_mode.

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
        
    def get_obs(self):
        """Get the observations of the systems. 
        (History) Hisotrics of observations can be added. The order of how
        it is added makes no difference for SB3 RL algorithm."""
        # Large scalling to have precise knowledge of observers. 
        state = []

        y = self.y

        # Time t
        if self.obs_time:
            state.append(self._get_time_obs())

        # Ovitraps E
        if self.obs_E:
            e_obs = y[0]

            if self.obs_multi_scaling:
                state.append(normalize(min(e_obs, 10 * self.K), 0, 10 * self.K))
                state.append(normalize(min(e_obs, self.K),      0, self.K))
                state.append(normalize(min(e_obs, 5000),        0, 5000))
                state.append(normalize(min(e_obs, 500),         0, 500))
                state.append(normalize(min(e_obs, 50),          0, 50))
                state.append(normalize(min(e_obs, 5),           0, 5))
            else:
                state.append(normalize(y[0], 0, 2 * self.K))

        return np.array(state) 
    
    def reward(self, action=None):
        reward = 0.0
        reward_info = {}

        # penalize norm of first three observer errors
        if self.action_only_E :
            diff_state_obs = self.y[0] - self.y_hat 
            
            r = -self.rwd_y1 * np.linalg.norm(diff_state_obs) / self.K
            reward_info["rwd_y1"] = r
            reward += r
        else:
            diff_state_obs = self.y - self.y_hat
            if self.rwd_y123 > 0:
                r = -self.rwd_y123 * np.linalg.norm(diff_state_obs) / self.K
                reward_info["rwd_y123"] = r
                reward += r

        return reward, reward_info
    
    def norm_y(self):
        """Compute the norm of the state without the control u."""
        return np.sum(self.y**2)**0.5

    def R_0(self):
        '''Offspring number of the system'''
        return self.beta_E * self.nu_E* self.nu / (self.delta_F*(self.delta_E + self.nu_E))
    
    def eq_point(self):
        '''Non zero equilibrium point of the system'''

        E_1 = self.K * (1 - 1 / self.R_0())
        M_1 = (1 - self.nu) * self.nu_E * E_1 / self.delta_M
        F_1 = self.nu * self.nu_E * E_1 / self.delta_F

        return np.array([E_1, M_1, F_1])
    
    def plot(self, title="Mosquito model dynamics", eq_point = None, bool_show=True):
        '''Plot le résultat de la simulation du système.'''

        array_t = np.array(self.t_lst)
        array_Y = np.array(self.y_lst)   # shape (T, 3)

        # Couleurs définies une fois
        cE = 'tab:blue'
        cM = 'tab:orange'
        cF = 'tab:green'

        # --- Premier subplot : E, M, F ---
        plt.plot(array_t, array_Y[:, 0]/self.K, label='E(t)/K', color=cE)
        plt.plot(array_t, array_Y[:, 1]/self.K, label='M(t)/K', color=cM)
        plt.plot(array_t, array_Y[:, 2]/self.K, label='F(t)/K', color=cF)
        plt.title(title + ': E/K, M/K, F/K')
        plt.xlabel('Temps (jours)')
        plt.ylabel('Population')
        plt.legend()


        # Add the equilibrium point 
        if not(eq_point is None):
            plt.axhline(eq_point[0], linestyle='--', color=cE, label='E*')
            plt.axhline(eq_point[1], linestyle='--', color=cM, label='M*')
            plt.axhline(eq_point[2], linestyle='--', color=cF, label='F*')

        plt.tight_layout()

        if bool_show:
            plt.show()
            
        return 
