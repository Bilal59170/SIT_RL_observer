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


class SimODE_old:
    """Classe utilisée pour les premiers tests (rewards relative, construction d'observateur avec une dynamique..).
    On préfère reconstruire des classes plus simples (ODE_model_v1 pour une classe sans contrôle par exemple)
     et faire marcher les simulations dessus"""
    def __init__(
        self,
        u_types : dict,
        dt, 
        tmax, 

        dynamics_mode=False, # Observer constructed with its dynamics
        dynamics_z = False, # dynamical term in the reward
        mu_z = None,
        lambda_z = None, 
        dynamics_L=False, # Observer with a Luenberger like dynamics. 
    

        rwd_y123=0, 
        rwd_y4=0, 
        rwd_y123_pos=0,
        rwd_y4_pos=0,
        rwd_y_relat=0,
        threshold_y_relat=100, # We consider that 100 mosquitoes is very small: 100 << 50k = K
        factor_K = 1, 
        y0_without_u0=None,

        # Observations
        obs_y=False,
        obs_u=True,
        obs_E_state=False,
        obs_Ms=False, 

        y_period=None,
        length_obs_history :int = 1,
        period_random_ctrl = 7,
        min_value_obs=0, # minimal value authorized for the observer 
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
        
        self.min_value_obs = min_value_obs
        # Control parameters
        self.umax = 170e3 # U_max in Sota paper
        self.period_random_ctrl = period_random_ctrl
        self.u_types = u_types
        self.ctrl_mode_resurgence = 0 # set of the control is 0 or not to allow resurgence
        # Set the control function
        self.u_fct, self.current_u_type = choose_u(
            u_types, umax=self.umax, period_random_ctrl=period_random_ctrl, tmax=tmax, instance_ODE_model=self)
        

        # Simulation parameters
        self.dt    = dt
        self.tmax  = tmax
        self.rwd_y123 = rwd_y123
        self.rwd_y4   = rwd_y4
        self.rwd_y123_pos = rwd_y123_pos
        self.rwd_y4_pos   = rwd_y4_pos
        self.rwd_y_relat = rwd_y_relat
        self.threshold_y_relat = threshold_y_relat
        self.factor_K = factor_K

        # Observations
        self.obs_y = obs_y
        self.obs_u = obs_u
        self.obs_E_state = obs_E_state
        self.obs_Ms = obs_Ms

        # History for the whole state y
        self.length_obs_history = length_obs_history
        self._y_history = deque([np.zeros(5) for _ in range(length_obs_history)],
                                maxlen=length_obs_history) #(E, M, F, Ms, u)
        self.next_y_time = np.inf
        self.y_period = float(y_period) if y_period is not None else None # e.g 1 for 1 day

        # History for y_hat (same structure as y. For dynamical way of determining the observer)
        self._y_hat_history = deque([np.zeros(4) for _ in range(length_obs_history)],
                                    maxlen=length_obs_history)  # (E_hat, M_hat, F_hat, Ms_hat)
        self.last_y_hat_obs = np.zeros(4)
        self.next_y_hat_time = np.inf

        # Approximation of either the dynamics or the values
        self.dynamics_mode = dynamics_mode 
        self.dynamics_z = dynamics_z
        self.mu_z, self.lambda_z = mu_z, lambda_z
        if self.dynamics_z and (self.mu_z is None or self.lambda_z is None):
            raise ValueError(
                "dynamics_z=True requires mu_z and lambda_z")
        self.dynamics_L = dynamics_L

        # Shape of the action
        if self.dynamics_L:
            self.n_action = 16 # Matrice de gain L(t) 4x4
        else : 
            self.n_action = 4

        #### Reset of the simulation ####
        self.reset(y0_without_u0)

        
    def reset(self, y0_without_u0=None):
        '''Reset the simulation. Called at each episode.'''
        self.t = 0.0

        # Take another control for the next episode
        self.ctrl_mode_resurgence = 0 # Put back the control to inactive in the case of control for resurgence 
        self.u_fct, self.current_u_type = choose_u(
            self.u_types, umax=self.umax, period_random_ctrl=self.period_random_ctrl, tmax=self.tmax, instance_ODE_model=self)
        # Initialize (E, M, F, Ms)
        if y0_without_u0 is not None:
            if len(y0_without_u0)!=4:
                raise ValueError("y0_without_u0 should be a numpy array of shape (,4)")
            emfms = np.array(y0_without_u0, dtype=float)
        else:
            emfms = np.random.uniform(low=0.0, high= self.factor_K * self.K, size=(4))

        # Initialize u with u(E, M, F, Ms, t=0)
        self.y = np.array([*emfms, self.u_fct(t=0, EMFMs=emfms)]) # State of the system
        
        # Initialize randomly the observer that is constructed with reinforcement learning: (E, M, F, Ms) hat (no u)
        self.y_hat = np.random.uniform(low=0.0, high= self.factor_K * self.K, size=(4)) 

        # Initialize variable z for observer strategy y_hat_dot = f(y_hat, u) + z, z_dot = -lambda * z + mu * e
        
        if self.dynamics_z:
            self.z = 0.0
            self.z_lst = [np.copy(self.z)]


        self.t_lst = [self.t]
        self.y_lst = [np.copy(self.y)]
        self.y_hat_lst = [np.copy(self.y_hat)]

        # Full observation of the state y=(E, M, F, Ms, u)
        if self.obs_y : 
            self.last_y_obs = np.copy(self.y)
            self._y_history = deque([np.copy(self.y) for _ in range(self.length_obs_history)],
                                     maxlen=self.length_obs_history)

            if self.y_period is not None:
                self.next_y_time = self.y_period
            else:
                self.next_y_time = np.inf
        
        # Historique de y_hat (même structure que y)
        if self.dynamics_mode:
            self.last_y_hat_obs = np.copy(self.y_hat)
            self._y_hat_history = deque([np.copy(self.y_hat) for _ in range(self.length_obs_history)],
                                        maxlen=self.length_obs_history)
            if self.y_period is not None:
                self.next_y_hat_time = self.y_period
            else:
                self.next_y_hat_time = np.inf

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

    def update_y(self, action):
        '''One Euler step.'''
        # Computation of (E, M, F, Ms)(t+dt) with u(t)
        new_emfs = self.y[:4] + self.dt * self.dynamics(self.y)

        # u(t+dt) = k(t, (E, F, M, Ms)(t)) 
        new_u    = self.u_fct(self.t, self.y[:4])

        # Rebuild full state (E, M, F, Ms, u)(t+dt)
        self.y   = np.append(new_emfs, new_u)

        #### Update the observer ####
        if not(self.dynamics_mode): # Direct approximation of the state : x_hat = action 
            self.y_hat = action
        else: # Approximate the derivate dx_hat/dt: The derivative is of time step dt_a := nb_step_per_action * dt 
            if self.dynamics_L:
                L = action.reshape(4, 4)  # matrice de gain 4x4
                e = (self.y_hat - self.y[:4]) / self.K   # erreur d'observation, shape (4,)
                y_hat_dot = self.dynamics(np.array([*self.y_hat, self.y[4]])) - L @ e
                self.y_hat = np.maximum(self.y_hat + self.dt * y_hat_dot, self.min_value_obs)

            elif self.dynamics_z:
                # Full observation case, i.e C = Id_4. e := obs - Cy_hat, C observation matrix in the linear case
                e_scalar = self.rwd_y123 * np.linalg.norm(
                        [self.y[0]-self.y_hat[0], self.y[1]-self.y_hat[1], self.y[2]-self.y_hat[2]]
                        ) / self.K + self.rwd_y4 * abs(self.y[3]-self.y_hat[3]) / self.K
                
                z_dot = -self.lambda_z * self.z + self.mu_z * e_scalar
                self.z += self.dt * z_dot
                self.z = np.clip(self.z, -1, 1)

                self.y_hat = action
            else : 
                self.y_hat = np.maximum(self.y_hat + self.dt * action, self.min_value_obs) # take positive part to avoid negative states

        # Update the histories of the observations
        if self.obs_y:
            self._update_y_observation()
        if self.dynamics_mode:
            self._update_y_hat_observation()

    def step(self, action, store=False):
        """Execute one step of the simulation.

        store: if True, append (t, y) to t_lst/y_lst for later plotting.
               Keep False during training to avoid memory growth on long
               episodes (100k+ steps per episode).
        """
        # Increment time
        self.t += self.dt
        self.update_y(action)

        # save for later plotting/metrics only when requested
        if store:
            self.t_lst.append(self.t)
            self.y_lst.append(np.copy(self.y))
            self.y_hat_lst.append(np.copy(self.y_hat))
            if self.dynamics_z: self.z_lst.append(np.copy(self.z))

    def simulate(self, n_steps):
        """Simulates the dynamics of the system. Used in tests to see the chosen control acting."""
        for _ in range(n_steps):
            self.step(action=np.zeros(self.n_action), store=True) # action = np.zeros(self.n_action) : fake action to only focus on real state

    def _update_y_observation(self):
        """Capteur de comptage de tout l'état y=(E, M, F, Ms, u).

        - y_period None  : lecture instantanee a chaque appel.
        - y_period defini : releve seulement tous les y_period jours ;
          entre deux releves, l'agent voit la derniere valeur connue.
        """
        if self.y_period is None:
            # comportement instantane (retrocompatible avec le cas sans historique)
            self.last_y_obs = np.copy(self.y)
            self._y_history.append(self.last_y_obs)
            return 
        
        eps = 1e-12
        while self.t + eps >= self.next_y_time:
            self.last_y_obs = np.copy(self.y)
            self._y_history.append(self.last_y_obs)
            self.next_y_time += self.y_period
        
    def _update_y_hat_observation(self):
        """Historique de y_hat, même logique que _update_y_observation()."""
        if self.y_period is None:
            self.last_y_hat_obs = np.copy(self.y_hat)
            self._y_hat_history.append(self.last_y_hat_obs)
            return

        eps = 1e-12
        while self.t + eps >= self.next_y_hat_time:
            self.last_y_hat_obs = np.copy(self.y_hat)
            self._y_hat_history.append(self.last_y_hat_obs)
            self.next_y_hat_time += self.y_period

    def get_obs(self):
        # Large scalling to have precise knowledge of observers. More precision than the control case is maybe needed 
        state = []
        y = self.y

        if self.dynamics_z:
            state.append(self.z)

        # E_hat
        for y_hat_hist in self._y_hat_history:
            e = y_hat_hist[0]
            state.append(normalize(e, 0, 10 * self.K))
            state.append(normalize(e, 0, self.K))
            state.append(normalize(min(e, 5000), 0, 5000))
            state.append(normalize(min(e, 500),  0, 500))
            state.append(normalize(min(e, 50),   0, 50))
            state.append(normalize(min(e, 5),    0, 5))
        # M_hat
        for y_hat_hist in self._y_hat_history:
            m = y_hat_hist[1]
            state.append(normalize(m, 0, 10 * self.K))
            state.append(normalize(m, 0, self.K))
            state.append(normalize(min(m, 5000), 0, 5000))
            state.append(normalize(min(m, 500),  0, 500))
            state.append(normalize(min(m, 50),   0, 50))
            state.append(normalize(min(m, 5),    0, 5))
        # F_hat
        for y_hat_hist in self._y_hat_history:
            f = y_hat_hist[2]
            state.append(normalize(f, 0, 10 * self.K))
            state.append(normalize(f, 0, self.K))
            state.append(normalize(min(f, 5000), 0, 5000))
            state.append(normalize(min(f, 500),  0, 500))
            state.append(normalize(min(f, 50),   0, 50))
            state.append(normalize(min(f, 5),    0, 5))
        # Ms_hat
        for y_hat_hist in self._y_hat_history:
            ms = y_hat_hist[3]
            state.append(normalize(ms, 0, 100 * self.K))
            state.append(normalize(min(ms, 50 * self.K), 0, 50 * self.K))
            state.append(normalize(min(ms, 10 * self.K), 0, 10 * self.K))
            state.append(normalize(min(ms, self.K),      0, self.K))
            state.append(normalize(min(ms, 5000),        0, 5000))
            state.append(normalize(min(ms, 500),         0, 500))
            state.append(normalize(min(ms, 50),          0, 50))
            state.append(normalize(min(ms, 5),           0, 5))

                
        if self.obs_y:
            # E
            for y_hist in self._y_history:
                # E
                e = y_hist[0]
                state.append(normalize(e, 0, 10 * self.K))
                state.append(normalize(e, 0, self.K))
                state.append(normalize(min(e, 5000), 0, 5000))
                state.append(normalize(min(e, 500),  0, 500))
                state.append(normalize(min(e, 50),   0, 50))
                state.append(normalize(min(e, 5),    0, 5))
            # M
            for y_hist in self._y_history:
                m = y_hist[1]
                state.append(normalize(m, 0, 10 * self.K))
                state.append(normalize(m, 0, self.K))
                state.append(normalize(min(m, 5000), 0, 5000))
                state.append(normalize(min(m, 500),  0, 500))
                state.append(normalize(min(m, 50),   0, 50))
                state.append(normalize(min(m, 5),    0, 5))
            # F
            for y_hist in self._y_history:
                f = y_hist[2]
                state.append(normalize(f, 0, 10 * self.K))
                state.append(normalize(f, 0, self.K))
                state.append(normalize(min(f, 5000), 0, 5000))
                state.append(normalize(min(f, 500),  0, 500))
                state.append(normalize(min(f, 50),   0, 50))
                state.append(normalize(min(f, 5),    0, 5))
            # Ms
            for y_hist in self._y_history:
                ms = y_hist[3]
                state.append(normalize(ms, 0, 100 * self.K))
                state.append(normalize(min(ms, 50 * self.K), 0, 50 * self.K))
                state.append(normalize(min(ms, 10 * self.K), 0, 10 * self.K))
                state.append(normalize(min(ms, self.K),      0, self.K))
                state.append(normalize(min(ms, 5000),        0, 5000))
                state.append(normalize(min(ms, 500),         0, 500))
                state.append(normalize(min(ms, 50),          0, 50))
                state.append(normalize(min(ms, 5),           0, 5))
            # u
            for y_hist in self._y_history:
                u_obs = y_hist[4]
                state.append(normalize(min(u_obs, 10 * self.K), 0, 10 * self.K)) # ~ delta_s * Ms
                state.append(normalize(min(u_obs, self.K),      0, self.K))
                state.append(normalize(min(u_obs, 5000),        0, 5000))
                state.append(normalize(min(u_obs, 500),         0, 500))
                state.append(normalize(min(u_obs, 50),          0, 50))
                state.append(normalize(min(u_obs, 5),           0, 5))

        # Ovitraps E
        if self.obs_E_state:
            # state.append(normalize(y[0], 0, 2 * self.K))
            e_obs = y[0]
            state.append(normalize(min(e_obs, 10 * self.K), 0, 10 * self.K))
            state.append(normalize(min(e_obs, self.K),      0, self.K))
            state.append(normalize(min(e_obs, 5000),        0, 5000))
            state.append(normalize(min(e_obs, 500),         0, 500))
            state.append(normalize(min(e_obs, 50),          0, 50))
            state.append(normalize(min(e_obs, 5),           0, 5))

        # Mâles stériles Ms
        if self.obs_Ms:
            # state.append(normalize(y[3], 0, 50 * self.K))

            # Same rescaling as M+Ms in the control code
            ms = y[3]
            state.append(normalize(ms, 0, 100 * self.K))
            state.append(normalize(min(ms, 50 * self.K), 0, 50 * self.K))
            state.append(normalize(min(ms, 10 * self.K), 0, 10 * self.K))
            state.append(normalize(min(ms, self.K),      0, self.K))
            state.append(normalize(min(ms, 5000),        0, 5000))
            state.append(normalize(min(ms, 500),         0, 500))
            state.append(normalize(min(ms, 50),          0, 50))
            state.append(normalize(min(ms, 5),           0, 5))

        # control u
        if self.obs_u :
            # state.append(normalize(y[4], 0, 2 * self.K))
            
            u_obs = y[4]
            state.append(normalize(min(u_obs, 10 * self.K), 0, 10 * self.K)) # ~ delta_s * Ms
            state.append(normalize(min(u_obs, self.K),      0, self.K))
            state.append(normalize(min(u_obs, 5000),        0, 5000))
            state.append(normalize(min(u_obs, 500),         0, 500))
            state.append(normalize(min(u_obs, 50),          0, 50))
            state.append(normalize(min(u_obs, 5),           0, 5))

        return np.array(state)  # no normalisation or multi-scale strategy for now
    
    def reward(self, action=None):
        reward = 0
        reward_info = {}

        # penalize norm of first three observer errors
        if self.rwd_y123 > 0:
            rwd_y123 = -self.rwd_y123 * np.linalg.norm(
                [self.y[0]-self.y_hat[0], self.y[1]-self.y_hat[1], self.y[2]-self.y_hat[2]]
                ) / self.K
            reward_info["rwd_y123"] = rwd_y123
            reward += rwd_y123

        # penalize (norm of) fourth observer errors (M_s)
        if self.rwd_y4 > 0:
            rwd_y4 = -self.rwd_y4 * abs(self.y[3]-self.y_hat[3]) / self.K
            reward_info["rwd_y4"] = rwd_y4
            reward += rwd_y4

        # Pénalise lorsque l'observateur est plus bas que l'état. Pour atténuer le problème d'annulation de x_hat
        if self.rwd_y123_pos > 0:
            rwd_y123_pos = -self.rwd_y123_pos\
                * np.linalg.norm(np.array([max(self.y[i] - self.y_hat[i], 0) for i in range(3)]))\
                / self.K
            reward_info["rwd_y123_pos"] = rwd_y123_pos
            reward += rwd_y123_pos

        # Similaire au terme du dessus, pour Ms_hat
        if self.rwd_y4_pos > 0:
            rwd_y4_pos = -self.rwd_y4_pos * max(self.y[3] - self.y_hat[3], 0) / self.K
            reward_info["rwd_y4_pos"] = rwd_y4_pos
            reward += rwd_y4_pos
        
        # Calcule une récompense avec un écart relatif. Permet de ne pas introduire 
        # d'échelle caractéristique du système, comme K 
        if self.rwd_y_relat > 0:
            rwd_y_relat = -self.rwd_y_relat * np.linalg.norm(
                [(self.y[i] - self.y_hat[i]) / max(self.y[i], self.threshold_y_relat) for i in range(4)]
                )
            reward_info["rwd_y_relat"] = rwd_y_relat
            reward += rwd_y_relat
        
        if self.dynamics_z:
            reward -= self.z

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

    def plot_observer(self, title="Observer approximation for Mosquito model dynamics", title_ctrl = "", bool_show=True):
        '''Plot le résultat de la simulation du système sans contrôle.'''
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))


        array_t = np.array(self.t_lst)
        array_Y = np.array(self.y_lst)   # shape (T, 5)
        array_Y_hat = np.array(self.y_hat_lst)   # shape (T, 5)


        # Colours
        cx = 'tab:blue'
        chat = 'tab:red'

        # Subplot for E
        axes[0].plot(array_t, array_Y[:, 0], label='E(t)', color=cx) 
        axes[0].plot(array_t, array_Y_hat[:, 0], linestyle='--', label='E_hat(t)', color=chat)
        axes[0].set_title('E, E_hat')
        axes[0].set_xlabel('Temps (jours)')
        axes[0].set_ylabel('Population')
        axes[0].legend()

        # Subplot for M
        axes[1].plot(array_t, array_Y[:, 1], label='M(t)', color=cx)
        axes[1].plot(array_t, array_Y_hat[:, 1], linestyle='--', label='M_hat(t)', color=chat)
        axes[1].set_title('M, M_hat')
        axes[1].set_xlabel('Temps (jours)')
        axes[1].set_ylabel('Population')
        axes[1].legend()

        # Subplot for F
        axes[2].plot(array_t, array_Y[:, 2], label='F(t)', color=cx)
        axes[2].plot(array_t, array_Y_hat[:, 2], linestyle='--',label='F_hat(t)', color=chat)
        axes[2].set_title('F, F_hat')
        axes[2].set_xlabel('Temps (jours)')
        axes[2].set_ylabel('Population')
        axes[2].legend()

        # Subplot for Ms
        axes[3].plot(array_t, array_Y[:, 3], label='Ms(t)', color=cx)
        axes[3].plot(array_t, array_Y_hat[:, 3], linestyle='--',label='Ms_hat(t)', color=chat)
        axes[3].set_title('Ms, Ms_hat')
        axes[3].set_xlabel('Temps (jours)')
        axes[3].set_ylabel('Population')
        axes[3].legend()

        fig.suptitle(title)
        plt.tight_layout()

        if bool_show:
            plt.show()
            
        return fig, axes
    