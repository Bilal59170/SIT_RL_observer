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

        # Observer (attribute in SimODE_v2 ; not present in other classes
        # such as SimODE_old, with "direct" observer)
        observer_mode = getattr(self.sim, "observer_mode", "direct")
        self.is_luenberger_mode = observer_mode in ("luenberger_full", "luenberger_partial")

        # on_time_reward is meaningfull when observation_mode = "direct" : we can compare 
        # y_hat(t)= action to y(t) before carrying on in the simulation. In Luenberger mode, 
        # the effet an xhat of L has effect after the integration step so on-time-reward is useless in this case
        if self.on_time_reward and self.is_luenberger_mode:
            raise ValueError(
                f"on_time_reward=True est incompatible avec observer_mode={observer_mode!r} : "
                "l'action code un gain L, pas une estimation instantanée de l'état."
            )

        # observation space (the historic is managed by self.sim.get_obs())
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
                    # Bound if L(t) coefs. Arbitraty values advised by Claude to test. 
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
        return self.sim.reward() # action is not used in the reward, unlike the control case

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

        if self.on_time_reward: # appropriate to action = y_hat (mode "direct" uniquement
            self.sim.update_y_hat(action) # y_hat = action based on obs(t). Needed to compare y_hat(t) and y(t) in the reward

            #  r_t := f(y(t), y_hat(t)), y_hat mise à jour par le RL à t
            reward, reward_info = self.compute_reward() 

            # Steps of the simulation : y goes towards y(t+dt_a), dt_a = nb_step_per_action * dt
            for _ in range(self.n_steps_per_action):
                self.sim.step(action)

            state = self.get_state() # update obs: obs based on y(t+dt_a)

        else:
            for _ in range(self.n_steps_per_action):
                self.sim.step(action) # update the observer and system

            # update state
            state = self.get_state()

            #  r_t+dt_a := f(y(t+dt_a), y_hat(t+dt_a)), y_hat updated by RL at t and not t+dt_a
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

        # Security clip, complementary of the one of xhat in SimODE_v2) :
        # error raised by SB3/PyTorch if observation or reward have a NaN/Inf.
        if not np.all(np.isfinite(state)) or not np.isfinite(reward):
            terminated = True
            reward = -100.0
            info["nan_or_inf_explosion"] = True
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        return state, reward, terminated, truncated, info # obs_k+1, r_k

    def run_episode(self, model, deterministic=True, max_steps=None, store=True, tmax=None):
        """
        Plotting function similar to step method. Used for evaluation and callback. Store the history
        of y_hat and y to plot it.
        Args:
            model:self.model in callback
            deterministic: input of model.predict
            max_steps: max number of action step (None -> until done)
            store: store the dynamics in sim.t_lst/y_lst/y_hat_lst

        Returns:
            dict: t, Y, Y_hat (trajectories), rewards, t_rewards
            (Titime before every each action step), reward_infos, terminated, truncated
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
                # Same as RL_env.step() : compare y_hat(t) to y(t), before time step 
                # (only "direct" observer mode, cf. assert in ODE_model_v2.py init)
                self.sim.update_y_hat(action)
                reward, reward_info = self.compute_reward()

                rewards.append(reward)
                t_rewards.append(self.sim.t)
                for k, v in reward_info.items():
                    reward_infos[k].append(v)

                for _ in range(self.n_steps_per_action):
                    self.sim.step(action, store=store)

                state = self.get_state()
            else:
                for _ in range(self.n_steps_per_action):
                    self.sim.step(action, store=store)

                state = self.get_state()

                reward, reward_info = self.compute_reward()
                rewards.append(reward)
                t_rewards.append(self.sim.t)
                for k, v in reward_info.items():
                    reward_infos[k].append(v)

            terminated = self.sim.norm_y() > 1000 * self.sim.K   # explosion
            truncated = self.sim.t >= tmax                  # time limit

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