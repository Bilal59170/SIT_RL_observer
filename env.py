from abc import ABC
import numpy as np
import gymnasium as gym
from models.ODE_model_old import SimODE_old

from collections import defaultdict


class RL_env(gym.Env, ABC):
    def __init__(self, sim_class, sim_kwargs, config):
        super().__init__()

        # load config
        self.config = config
        self.tmax = config["tmax"]
        self.n_steps_per_action = config["n_steps_per_action"]
        self.on_time_reward = config["on_time_reward"]

        # create simulation
        # self.sim = sim_class(**sim_kwargs, tmax=self.tmax)
        self.sim = sim_class(**sim_kwargs)
        self.sim.reset()

        # observation space (l'historique, s'il y en a, est géré par self.sim.get_obs())
        self.n_observations = self.sim.get_obs().shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_observations,), dtype=np.float32,)

        # action space — self.sim.n_action observers
        self.n_actions = self.sim.n_action
        self.action_min = np.array(eval(config["action_min"]))
        self.action_max = np.array(eval(config["action_max"]))

        if isinstance(self.sim, SimODE_old):
            if self.sim.dynamics_mode:
                if self.sim.dynamics_L:
                    # Bornes sur les coefficients de L(t). Valeur arbitraires conseillées par Claude pour tester,
                    #  non vérifiées théoriquement.
                    val_action = 1.0
                    self.action_min = np.full(16, -val_action)
                    self.action_max = np.full(16,  val_action)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)

        # Actions shape check
        assert len(self.action_min) == self.n_actions, \
        f"action_min shape {len(self.action_min)} != n_actions {self.n_actions}"
        assert len(self.action_max) == self.n_actions, \
        f"action_max shape {len(self.action_max)} != n_actions {self.n_actions}"

        # reset env : sees the state of the system at time t=0 
        self.reset()

    def compute_reward(self, action=None):
        return self.sim.reward() # Action n'est pas utilisée dans la reward contrairement au cas du contrôle

    def get_state(self):
        '''Renvoit l'observation du système (historique éventuel géré par self.sim).'''
        return self.sim.get_obs()

    def normalize_action(self, action):
        '''Normalize the action from [-1, 1] -> [action_min, action_max]'''
        return (action + 1.0) * (self.action_max - self.action_min) / 2.0 + self.action_min

    # overridden method of gym.env
    def step(self, action):
        """Exécute un épisode complet"""

        # Normalize the action from [-1, 1] -> [action_min, action_max]
        action = self.normalize_action(action)

        if self.on_time_reward: # appropriate to action = y_hat
            self.sim.update_y_hat(action) # y_hat = action based on obs(t). Needed to compare y_hat(t) and y(t) in the reward

            #  r_t := f(y(t), y_hat(t)), y_hat mise à jour par le RL à t
            reward, reward_info = self.compute_reward() 

            # Steps of the simulation : y goes towards y(t+dt_a), dt_a = nb_step_per_action * dt
            for _ in range(self.n_steps_per_action):
                self.sim.step(action)

            state = self.get_state() # update obs: obs based on y(t+dt_a)

        else: # appropriate to aproximations of the dynamics of y_hat, e.g action = L with L Luenberger gain
            # Steps of the simulation : y goes towards y(t+dt_a), dt_a = nb_step_per_action * dt
            for _ in range(self.n_steps_per_action):
                self.sim.step(action) # update the observer and system

            # update state
            state = self.get_state()

            #  r_t+dt_a := f(y(t+dt_a), y_hat(t+dt_a)), y_hat mise à jour par le RL à t et non t+dt_a
            reward, reward_info = self.compute_reward()

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

        return state, reward, terminated, truncated, info # obs_k+1, r_k

    def run_episode(self, model, deterministic=True, max_steps=None, store=True, tmax=None):
        """
        Pareil que step, mais utilisé pour l'évaluation et le callback afin de garder
        en mémoire l'historique de tout l'épisode pour l'afficher.
        Similaire à self.step(): le récompense à t=T temps final n'est pas calculée.
        Args:
            model: objet avec .predict(obs, deterministic=...)
                (self.model dans le callback, ou un modèle worker en éval parallèle)
            deterministic: passé à model.predict
            max_steps: nb max de pas d'action (None = jusqu'à done)
            store: si True, stocke les sous-pas dans sim.t_lst/y_lst/y_hat_lst

        Returns:
            dict: t, Y, Y_hat (trajectoires complètes), rewards, t_rewards
            (temps AVANT chaque pas d'action), reward_infos, terminated, truncated
        """
        obs, _ = self.reset()
        tmax = tmax if tmax is not None else self.tmax

        state = obs
        done = False
        step = 0
        rewards, t_rewards = [], []
        reward_infos = defaultdict(list)
        terminated = truncated = False

        while not done and (max_steps is None or step < max_steps):
            action, _ = model.predict(state, deterministic=deterministic)
            action = self.normalize_action(action)
            

            if self.on_time_reward:
                # --- aligné sur RL_env.step() : comparer y_hat(t) à y(t), AVANT d'avancer ---
                self.sim.update_y_hat(action)
                reward, reward_info = self.compute_reward()

                rewards.append(reward)
                t_rewards.append(self.sim.t)
                for k, v in reward_info.items():
                    reward_infos[k].append(v)

                # --- avance la sim   ---
                for _ in range(self.n_steps_per_action):
                    self.sim.step(action, store=store)

                state = self.get_state()
            else:
                # --- avance la sim   ---
                for _ in range(self.n_steps_per_action):
                    self.sim.step(action, store=store)

                state = self.get_state()

                reward, reward_info = self.compute_reward()
                rewards.append(reward)
                t_rewards.append(self.sim.t)
                for k, v in reward_info.items():
                    reward_infos[k].append(v)

            terminated = self.sim.norm_y() > 1000 * self.sim.K   # explosion
            truncated = self.sim.t >= tmax                  # limite de temps

            if terminated:
                rewards[-1] -= 100

            done = terminated or truncated
            step += 1
        return {
            "t": np.array(self.sim.t_lst),
            "Y": np.array(self.sim.y_lst),
            "Y_hat": np.array(self.sim.y_hat_lst),
            "rewards": np.array(rewards),
            "t_rewards": np.array(t_rewards),
            "reward_infos": dict(reward_infos),
            "terminated": terminated,
            "truncated": truncated,
    }

    def render(self):
        # render not implemented (avoid SB3 warnings)
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # initialise the random variable generator of the rest sim

        if seed is not None:
            np.random.seed(seed)

        # reset simulation
        self.sim.reset()

        # get initial state
        s0 = self.get_state()

        return s0, {} # Returns (obs, info), with info being a dict