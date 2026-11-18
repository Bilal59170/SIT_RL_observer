from abc import ABC
import numpy as np
import gymnasium as gym
from models.ODE_model_old import SimODE_old


class RL_env(gym.Env, ABC):
    def __init__(self, sim_class, sim_kwargs, config):
        super().__init__()

        # load config
        self.config = config
        self.tmax = config["tmax"]
        self.n_steps_per_action = config["n_steps_per_action"]

        # create simulation
        self.sim = sim_class(**sim_kwargs, tmax=self.tmax)
        self.sim.reset()

        # memory
        self.n_observations_base = self.sim.get_obs().shape[0]
        self.n_past_states = config["n_past_states"]
        self.mem_default_value = 0.0
        self.reset_memory()

        # observation space
        self.n_observations = self.n_observations_base * (1 + self.n_past_states) # Current + past observations
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_observations,), dtype=np.float32,)
        
        # action space — self.sim.n_action observers 
        self.n_actions = self.sim.n_action
        self.action_min = np.array(eval(config["action_min"])) 
        self.action_max = np.array(eval(config["action_max"])) 

        
        if isinstance(self.sim, SimODE_old): 
            if self.sim.dynamics_mode:
                # if self.sim.dynamics_L and (self.action_min.size != 2  or self.action_max.size != 2):
                #     raise ValueError("Shape of the action should be (1, 2)")
                if self.sim.dynamics_L:
                    # Bornes sur les coefficients de L(t). Valeur arbitraires conseillées par Claude pour tester,
                    #  non vérifiées théoriquement.
                    val_action = 1.0
                    self.action_min = np.full(16, -val_action) 
                    self.action_max = np.full(16,  val_action)
                    # print(f"[RL_env] dynamics_L: action_min/max overridden to ±{val_action}")
        # else:
        #     if not(isinstance(self.sim, SimODE_v1)) and not(isinstance(self.sim, SimODE_v2)):
        #         raise ValueError("Error in the Class type : must be either SimODE_old or SimODE_v")
                
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)

        # Actions shape check
        assert len(self.action_min) == self.n_actions, \
        f"action_min shape {len(self.action_min)} != n_actions {self.n_actions}"
        assert len(self.action_max) == self.n_actions, \
        f"action_max shape {len(self.action_max)} != n_actions {self.n_actions}"
        
        # reset env
        self.reset()

    def compute_reward(self, action):
        return self.sim.reward(action)

    def get_base_state(self):
        '''Renvoit l'observation du système'''
        return self.sim.get_obs()

    def reset_memory(self):
        '''Reset memory with constant vector of value self.mem_default_value = 0 by default.'''
        self.memory = np.full(shape = self.n_observations_base * self.n_past_states,
                              fill_value=self.mem_default_value, dtype=np.float32)

    def get_state(self, append_to_memory=False):
        '''Returns the observations of the systems. Update historic if needed.'''
        base_state = self.get_base_state()
        state = np.concatenate((base_state, self.memory))

        if append_to_memory and self.n_past_states > 0:
            self.memory = np.roll(self.memory, self.n_observations_base)
            self.memory[: self.n_observations_base] = base_state
        return state
    
    def normalize_action(self, action):
        '''Normalize the action from [-1, 1] -> [action_min, action_max]'''
        return (action + 1.0) * (self.action_max - self.action_min) / 2.0 + self.action_min
    
    # overridden method of gym.env
    def step(self, action):
        # Normalize the action from [-1, 1] -> [action_min, action_max]
        action = self.normalize_action(action)

        # step simulation with control
        for _ in range(self.n_steps_per_action):
            self.sim.step(action) # update the observer 

        # update state
        state = self.get_state(append_to_memory=True)

        # compute reward
        reward, reward_info = self.compute_reward(action)

        
        info = {
            'scaled_action': action,
            'reward_info': reward_info,}

        # API gymnasium : terminated (raison interne) vs truncated (limite temps)
        terminated = False
        truncated = False

        if self.sim.t >= self.tmax:
            truncated = True
            info["TimeLimit.truncated"] = True

        if self.sim.norm_y() > 1000 * self.sim.K:
            terminated = True
            reward -= 100
            info["explosion"] = True

        return state, reward, terminated, truncated, info

    def render(self):
        # render not implemented (avoid SB3 warnings)
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # initialise the random variable generator of the rest sim

        if seed is not None:
            np.random.seed(seed)

        # reset simulation and memory
        self.sim.reset() 
        self.reset_memory() 

        # get initial state
        s0 = self.get_state(append_to_memory=True)

        return s0, {} # Returns (obs, info), with info being a dict
