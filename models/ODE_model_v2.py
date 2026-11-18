"""
Simulating
    y'(t) = f(y(t), u(t)) with y(t) = (E(t), M(t), F(t), M_s(t))

Modified mosquito model:
    dE/dt   = beta_E * F * (1 - E/K) * M/(M + gamma * M_s) - (nu_E + delta_E) * E
    dF/dt   = nu * nu_E * E - delta_F * F
    dM/dt   = (1 - nu) * nu_E * E - delta_M * M
    dM_s/dt = u - delta_S * M_s

Main observation (ovitrap collection): E(t_n)

"""

import numpy as np
import matplotlib.pyplot as plt

from config import *
from collections import deque

def normalize(x, xmin, xmax):
    """Transform x from [xmin, xmax] range to [-1, 1] range."""
    return (x - xmin) / (xmax - xmin) * 2.0 - 1.0


class SimODE_v2:
    """Classe de simulation pour construire un observateur avec du Reinforcement Learning 
    avec plusieurs type de contrôles différents. Version moins lourde de ODE_model_old.py sans
      les features infructueuses pendant les entraînements comme la reward relative rwd_y_relat."""
    def __init__(
        self,
        u_types : dict,
        dt, 
        tmax, 
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
        obs_multi_scaling = True, # Observation scaling

        # Action
        action_only_E=False, # Output
        action_only_EMF=False,
        min_value_obs=0 # minimal value authorized for the observer 
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
        self.obs_multi_scaling = obs_multi_scaling

        # State
        self.n_state = 4

        # Action
        self.action_only_E = action_only_E
        self.action_only_EMF = action_only_EMF
        if self.action_only_E:
            self.n_action = 1 # Only E_hat to output
        elif self.action_only_EMF:
            self.n_action = 3
        else:
            self.n_action = 4
        self.min_value_obs = min_value_obs

        # Control parameters
        self.u_types = u_types
        self.ctrl_mode_resurgence = 0 # set of the control is 0 or not to allow resurgence
        # Set the control function
        self.u_fct, self.current_u_type = choose_u(
            u_types, tmax=tmax, instance_ODE_model=self)

        #### Reset of the simulation ####
        self.reset(y0_without_u0)

        
    def reset(self, y0_without_u0=None):
        '''Reset the simulation. Called at each episode.'''
        self.t = 0.0

        # Take another control for the next episode
        self.ctrl_mode_resurgence = 0 # Put back the control to inactive in the case of control for resurgence 
        self.u_fct, self.current_u_type = choose_u(
            self.u_types, tmax=self.tmax, instance_ODE_model=self)
        
        # Initializes (E, M, F, Ms)
        if y0_without_u0 is not None:
            if len(y0_without_u0)!=4:
                raise ValueError(f"y0_without_u0 should be a numpy array of shape (,{self.n_state})")
            emfms = np.array(y0_without_u0, dtype=float)
        else:
            emfms = np.random.uniform(low=0.0, high= self.factor_K * self.K, size=(self.n_state))

        # Initializes u with u(E, M, F, Ms, t=0)
        self.y = np.array([*emfms, self.u_fct(t=0, EMFMs=emfms)]) # full state of the system
        
        # Initializes randomly the observer that is constructed with reinforcement learning: (E, M, F, Ms) hat or (E, M, F) or (E)
        self.y_hat = np.random.uniform(low=0.0, high= self.factor_K * self.K, size=(self.n_action)) 

        self.t_lst = [self.t]
        self.y_lst = [np.copy(self.y)]
        self.y_hat_lst = [np.copy(self.y_hat)]

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
    
    def update_y(self):
        '''One Euler step of the system.'''
        # Computation of (E, M, F, Ms)(t+dt) with with explicit Euler
        new_emfs = self.y[:4] + self.dt * self.dynamics(self.y)

        # Computation of u(t+dt) = k(t+dt, y(t+dt))  
        new_u    = self.u_fct(self.t, new_emfs) 

        # Rebuild full state (E, M, F, Ms, u)(t+dt)
        self.y   = np.append(new_emfs, new_u)

    def update_y_hat(self, action): 
        """Update the observer. Used in env.py, not in step."""
        self.y_hat = action # Direct approximation of the state : y_hat = action 

    def step(self, action, store=False):
        """Execute one step of the simulation.

        store: if True, append (t, y) to t_lst/y_lst for later plotting.
               Keep False during training to avoid memory growth on long
               episodes (100k+ steps per episode).
        """

        # increment time
        self.t += self.dt
        self.update_y()
        self.update_y_hat(action)  # ← y_hat mis à jour à chaque pas

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
        (History) Historics of observations can be added. The order of how
        it is added makes no difference for SB3 RL algorithm."""
        # Large scalling to have precise knowledge of observers. 
        state = []

        y = self.y
        y_hat = self.y_hat

        # Time t
        if self.obs_time:
            state.append(self._get_time_obs())

        # Observer
        if self.obs_y_hat: 
            if self.action_only_E:
                e_hat = y_hat[0]

                if self.obs_multi_scaling:
                    state.append(normalize(min(e_hat, 10 * self.K), 0, 10 * self.K))
                    state.append(normalize(min(e_hat, self.K),      0, self.K))
                    state.append(normalize(min(e_hat, 5000),        0, 5000))
                    state.append(normalize(min(e_hat, 500),         0, 500))
                    state.append(normalize(min(e_hat, 50),          0, 50))
                    state.append(normalize(min(e_hat, 5),           0, 5))
                else:
                    state.append(normalize(e_hat, 0, 2 * self.K))
            elif self.action_only_EMF:
                e_hat , m_hat , f_hat = y_hat[0], y_hat[1], y_hat[2]

                if self.obs_multi_scaling:
                    state.append(normalize(min(e_hat , 10 * self.K), 0, 10 * self.K))
                    state.append(normalize(min(e_hat , self.K),      0, self.K))
                    state.append(normalize(min(e_hat , 5000),        0, 5000))
                    state.append(normalize(min(e_hat , 500),         0, 500))
                    state.append(normalize(min(e_hat , 50),          0, 50))
                    state.append(normalize(min(e_hat , 5),           0, 5))

                    state.append(normalize(min(m_hat , 10 * self.K), 0, 10 * self.K))
                    state.append(normalize(min(m_hat , self.K),      0, self.K))
                    state.append(normalize(min(m_hat , 5000),        0, 5000))
                    state.append(normalize(min(m_hat , 500),         0, 500))
                    state.append(normalize(min(m_hat , 50),          0, 50))
                    state.append(normalize(min(m_hat , 5),           0, 5))

                    state.append(normalize(min(f_hat, 10 * self.K), 0, 10 * self.K))
                    state.append(normalize(min(f_hat, self.K),      0, self.K))
                    state.append(normalize(min(f_hat, 5000),        0, 5000))
                    state.append(normalize(min(f_hat, 500),         0, 500))
                    state.append(normalize(min(f_hat, 50),          0, 50))
                    state.append(normalize(min(f_hat, 5),           0, 5))
                else:
                    state.append(normalize(e_hat , 0, 2 * self.K))
                    state.append(normalize(m_hat , 0, 2 * self.K))
                    state.append(normalize(f_hat, 0, 5 * self.K))
            else: # Case with  y_hat = (E_hat, M_hat, F_hat, Ms_hat)
                e_hat , m_hat , f_hat, ms_hat = y_hat[0], y_hat[1], y_hat[2], y_hat[3]

                if self.obs_multi_scaling:
                    state.append(normalize(min(e_hat , 10 * self.K), 0, 10 * self.K))
                    state.append(normalize(min(e_hat , self.K),      0, self.K))
                    state.append(normalize(min(e_hat , 5000),        0, 5000))
                    state.append(normalize(min(e_hat , 500),         0, 500))
                    state.append(normalize(min(e_hat , 50),          0, 50))
                    state.append(normalize(min(e_hat , 5),           0, 5))

                    state.append(normalize(min(m_hat , 10 * self.K), 0, 10 * self.K))
                    state.append(normalize(min(m_hat , self.K),      0, self.K))
                    state.append(normalize(min(m_hat , 5000),        0, 5000))
                    state.append(normalize(min(m_hat , 500),         0, 500))
                    state.append(normalize(min(m_hat , 50),          0, 50))
                    state.append(normalize(min(m_hat , 5),           0, 5))

                    state.append(normalize(min(f_hat, 10 * self.K), 0, 10 * self.K))
                    state.append(normalize(min(f_hat, self.K),      0, self.K))
                    state.append(normalize(min(f_hat, 5000),        0, 5000))
                    state.append(normalize(min(f_hat, 500),         0, 500))
                    state.append(normalize(min(f_hat, 50),          0, 50))
                    state.append(normalize(min(f_hat, 5),           0, 5))

                    state.append(normalize(ms_hat, 0, 100 * self.K))
                    state.append(normalize(min(ms_hat, 50 * self.K), 0, 50 * self.K))
                    state.append(normalize(min(ms_hat, 10 * self.K), 0, 10 * self.K))
                    state.append(normalize(min(ms_hat, self.K),      0, self.K))
                    state.append(normalize(min(ms_hat, 5000),        0, 5000))
                    state.append(normalize(min(ms_hat, 500),         0, 500))
                    state.append(normalize(min(ms_hat, 50),          0, 50))
                    state.append(normalize(min(ms_hat, 5),           0, 5))
                else:
                    state.append(normalize(e_hat , 0, 2 * self.K))
                    state.append(normalize(m_hat , 0, 2 * self.K))
                    state.append(normalize(f_hat, 0, 5 * self.K))
                    state.append(normalize(ms_hat, 0, 50 * self.K))
                

        # Ohter
        if self.obs_E_state:
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

        if self.obs_EMF:
            e_obs, m_obs, f_obs = y[0], y[1], y[2]

            if self.obs_multi_scaling:
                state.append(normalize(min(e_obs, 10 * self.K), 0, 10 * self.K))
                state.append(normalize(min(e_obs, self.K),      0, self.K))
                state.append(normalize(min(e_obs, 5000),        0, 5000))
                state.append(normalize(min(e_obs, 500),         0, 500))
                state.append(normalize(min(e_obs, 50),          0, 50))
                state.append(normalize(min(e_obs, 5),           0, 5))

                state.append(normalize(min(m_obs, 10 * self.K), 0, 10 * self.K))
                state.append(normalize(min(m_obs, self.K),      0, self.K))
                state.append(normalize(min(m_obs, 5000),        0, 5000))
                state.append(normalize(min(m_obs, 500),         0, 500))
                state.append(normalize(min(m_obs, 50),          0, 50))
                state.append(normalize(min(m_obs, 5),           0, 5))

                state.append(normalize(min(f_obs, 10 * self.K), 0, 10 * self.K))
                state.append(normalize(min(f_obs, self.K),      0, self.K))
                state.append(normalize(min(f_obs, 5000),        0, 5000))
                state.append(normalize(min(f_obs, 500),         0, 500))
                state.append(normalize(min(f_obs, 50),          0, 50))
                state.append(normalize(min(f_obs, 5),           0, 5))
            else:
                state.append(normalize(e_obs, 0, 2 * self.K))
                state.append(normalize(m_obs, 0, 2 * self.K))
                state.append(normalize(f_obs, 0, 5 * self.K))

        # Mâles stériles Ms
        if self.obs_Ms:
            # Same rescaling as M+Ms in the control code
            ms = y[3]
            if self.obs_multi_scaling:
                state.append(normalize(ms, 0, 100 * self.K))
                state.append(normalize(min(ms, 50 * self.K), 0, 50 * self.K))
                state.append(normalize(min(ms, 10 * self.K), 0, 10 * self.K))
                state.append(normalize(min(ms, self.K),      0, self.K))
                state.append(normalize(min(ms, 5000),        0, 5000))
                state.append(normalize(min(ms, 500),         0, 500))
                state.append(normalize(min(ms, 50),          0, 50))
                state.append(normalize(min(ms, 5),           0, 5))
            else:
                state.append(normalize(y[3], 0, 50 * self.K))

        # control u
        if self.obs_u :
            # state.append(normalize(y[4], 0, 2 * self.K))
            
            u_obs = y[4]
            if self.obs_multi_scaling:
                state.append(normalize(min(u_obs, 10 * self.K), 0, 10 * self.K)) # ~ delta_s * Ms
                state.append(normalize(min(u_obs, self.K),      0, self.K))
                state.append(normalize(min(u_obs, 5000),        0, 5000))
                state.append(normalize(min(u_obs, 500),         0, 500))
                state.append(normalize(min(u_obs, 50),          0, 50))
                state.append(normalize(min(u_obs, 5),           0, 5))
            else: 
                state.append(normalize(min(u_obs, 10 * self.K), 0, 10 * self.K))

        return np.array(state)  # no normalisation or multi-scale strategy for now
    
    def reward(self, action=None):
        reward = 0
        reward_info = {}

        # penalize norm of first three observer errors

        if self.rwd_y1 > 0:
            if not(self.action_only_E):
                raise ValueError("rwd_y1 needs only one action in E, i.e action_only_E=True")
            diff_state_obs = self.y[0] - self.y_hat[0]
            r = -self.rwd_y1 * np.linalg.norm(diff_state_obs) / self.K
            reward_info["rwd_y1"] = r
            reward += r

        if self.rwd_y123 > 0:
            if self.n_action < 3:
                raise ValueError("rwd_y123 requires at least 3 observed components (action_only_EMF or full mode)")
            diff_state_obs = self.y[:3] - self.y_hat[:3]
            r = -self.rwd_y123 * np.linalg.norm(diff_state_obs) / self.K
            reward_info["rwd_y123"] = r
            reward += r

        return reward, reward_info

    def norm_y(self):
        """Compute the norm of the state without the control u."""
        return np.sum(self.y[:-1]**2)**0.5

    def plot(self, title="Mosquito model dynamics", title_ctrl = "", eq_point = None, bool_show=True):
        '''Plot le résultat de la simulation du système sans contrôle.'''
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        array_t = np.array(self.t_lst)
        array_Y = np.array(self.y_lst)   # shape (T, 5)

        # Couleurs définies une fois
        cE = 'tab:blue'
        cM = 'tab:orange'
        cF = 'tab:green'
        cMs = 'tab:red'
        cu = 'tab:brown'

        # --- Premier subplot : E, M, F ---
        axes[0].plot(array_t, array_Y[:, 0]/self.K, label='E(t)/K', color=cE)
        axes[0].plot(array_t, array_Y[:, 1]/self.K, label='M(t)/K', color=cM)
        axes[0].plot(array_t, array_Y[:, 2]/self.K, label='F(t)/K', color=cF)
        axes[0].set_title('E/K, M/K, F/K')
        axes[0].set_xlabel('Temps (jours)')
        axes[0].set_ylabel('Population')
        axes[0].legend()

        # --- Deuxième subplot : Ms ---
        axes[1].plot(array_t, array_Y[:, 3]/self.K, label='Ms(t)', color=cMs)
        axes[1].set_title('Ms/K')
        axes[1].set_xlabel('Temps (jours)')
        axes[1].set_ylabel('Population')
        axes[1].legend()

        # --- Troisième subplot : u ---
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