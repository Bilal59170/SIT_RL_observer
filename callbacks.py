"""
callbacks.py — Callback tensorboard custom pour le modèle d'observateur
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure


import os
import time

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, run_id, every_seconds=5):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.run_id = run_id
        self.every_seconds = every_seconds

        self.start_time = None
        self.last_print_time = 0

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        now = time.time()

        # affichage toutes les X secondes
        if now - self.last_print_time < self.every_seconds:
            return True

        self.last_print_time = now

        elapsed = now - self.start_time
        steps = self.num_timesteps

        # progression (%)
        pct = 100.0 * steps / self.total_timesteps if self.total_timesteps > 0 else 0.0

        # vitesse + Temps restant
        if steps > 0 and elapsed > 0:
            speed = steps / elapsed  # steps/sec
            remaining_steps = self.total_timesteps - steps
            eta = remaining_steps / speed if speed > 0 else 0
        else:
            speed = 0
            eta = 0

        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        eta_s = int(eta % 60)

        print(
            f"[{self.run_id}] "
            f"[PID {os.getpid()}] " # "Process ID"
            f"{pct:6.2f}% "
            f"({steps}/{self.total_timesteps}) "
            f"| {speed:.0f} steps/s "
            f"| ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}", # "Estimated Time of Arrival"
            flush=True,
        )

        return True
    
    
class TensorboardCallback(BaseCallback):
    """Custom callback for plotting additional values in tensorboard."""

    def __init__(self, eval_env, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.eval_env = eval_env

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        pass

    def _on_rollout_end(self):
        try:
            # reset de l'environnement d'évaluation
            _, _ = self.eval_env.reset()

            # forcer le stockage du point initial
            self.eval_env.sim.t_lst = [self.eval_env.sim.t]
            self.eval_env.sim.y_lst = [np.copy(self.eval_env.sim.y)]
            self.eval_env.sim.y_hat_lst = [np.copy(self.eval_env.sim.y_hat)]

            gamma = self.model.gamma

            result = self.eval_env.run_episode(self.model, deterministic=True) # runs one episode of dt * nb_step_per_action steps
            rewards = result["rewards"]
            t_lst, y_lst, y_hat_lst = result["t"], result["Y"], result["Y_hat"]
            env_times = np.copy(result["t_rewards"])
            reward_infos = result["reward_infos"]

            rewards = np.array(rewards)
            t_lst = np.array(self.eval_env.sim.t_lst)
            y_lst = np.array(self.eval_env.sim.y_lst)         # shape (T, 5) : [E, M, F, Ms, u] 
            # shape (T, n_act), n_act in {1, 3, 4} : [E_hat, M_hat, F_hat, Ms_hat] or [E, M, F] or [E]
            y_hat_lst = np.array(self.eval_env.sim.y_hat_lst)  
            K = self.eval_env.sim.K

            # discounted reward gamma^t * r_t (t indexé sur les actions, pas le temps physique)
            discounted_rewards = np.array(
                [gamma**i * r for i, r in enumerate(rewards)]
            )

            # ── LOG SCALARS ──────────────────────────────────────────────
            self.logger.record('eval/cum_reward', np.sum(rewards))
            self.logger.record('eval/discounted_return', np.sum(discounted_rewards))

            # ── LOG PLOTS ────────────────────────────────────────────────
            def plot(name, x, y_dict, xlabel, ylabel, title,
                    grid=True, legend=True, figsize=(5, 3), max_points=1000):
                # sous-échantillonnage pour avoir un plot moins cher
                n = len(x)
                if n > max_points:
                    idx = np.linspace(0, n - 1, max_points, dtype=int)
                    x = np.array(x)[idx]
                    y_dict = {k: np.array(v)[idx] for k, v in y_dict.items()}

                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
                for k, v in y_dict.items():
                    ax.plot(x, v, label=k)
                if legend:
                    ax.legend()
                if grid:
                    ax.grid()
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                plt.tight_layout()
                self.logger.record(
                    f'eval/{name}', Figure(fig, close=True),
                    exclude=('stdout', 'log', 'json', 'csv')
                )
                plt.close()
            
            # 1. E, M, F (états réels)
            plot('EMF_by_t', t_lst,
                {'E/K': y_lst[:, 0]/K, 'M/K': y_lst[:, 1]/K, 'F/K': y_lst[:, 2]/K},
                xlabel='t', ylabel='Population', title='E/K, M/K, F/K')
                
            # Ms (état réel)
            plot('Ms_by_t', t_lst,
                {'Ms/K': y_lst[:, 3]/K},
                xlabel='t', ylabel='Population', title='Ms/K')
            
            # ── CHECK SHAPE y_hat_lst ─────────────────────────────────
            n_obs = y_hat_lst.shape[1]
            if n_obs not in (1, 3, 4):
                raise ValueError(f"y_hat_lst.shape[1] = {n_obs}, attendu 1, 3 ou 4.")

            # if not isinstance(self.eval_env.sim, SimODE_v1):
            #     raise ValueError("Should be SimODE_v1")

            # ── PLOTS ÉTATS RÉELS ─────────────────────────────────────
            if n_obs == 1:
                plot('E_by_t', t_lst,
                    {'E/K': y_lst[:, 0]/K},
                    xlabel='t', ylabel='Population', title='E/K')
            elif n_obs == 3:
                plot('EMF_by_t', t_lst,
                    {'E/K': y_lst[:, 0]/K, 'M/K': y_lst[:, 1]/K, 'F/K': y_lst[:, 2]/K},
                    xlabel='t', ylabel='Population', title='E/K, M/K, F/K')
            else:  # n_obs == 4
                plot('EMF_by_t', t_lst,
                    {'E/K': y_lst[:, 0]/K, 'M/K': y_lst[:, 1]/K, 'F/K': y_lst[:, 2]/K},
                    xlabel='t', ylabel='Population', title='E/K, M/K, F/K')

            # ── PLOTS OBSERVATEURS ET ERREURS ────────────────────────
            if n_obs == 1:
                plot('E_hat_by_t', t_lst,
                    {'E_hat/K': y_hat_lst[:, 0]/K},
                    xlabel='t', ylabel='Population estimée', title='E_hat/K')
                plot('errors_by_t', t_lst,
                    {'(E-E_hat)/K': (y_lst[:, 0] - y_hat_lst[:, 0])/K},
                    xlabel='t', ylabel='Erreur', title='Erreur d\'observation (E-E_hat)')

            elif n_obs == 3:
                plot('EMF_hat_by_t', t_lst,
                    {'E_hat/K': y_hat_lst[:, 0]/K, 'M_hat/K': y_hat_lst[:, 1]/K, 'F_hat/K': y_hat_lst[:, 2]/K},
                    xlabel='t', ylabel='Population estimée', title='E_hat/K, M_hat/K, F_hat/K')
                plot('errors_by_t', t_lst,
                    {'(E-E_hat)/K': (y_lst[:, 0] - y_hat_lst[:, 0])/K,
                     '(M-M_hat)/K': (y_lst[:, 1] - y_hat_lst[:, 1])/K,
                     '(F-F_hat)/K': (y_lst[:, 2] - y_hat_lst[:, 2])/K},
                    xlabel='t', ylabel='Erreur', title='Erreurs d\'observation (E, M, F)')
                plot('Ms_by_t', t_lst,
                    {'Ms/K': y_lst[:, 3]/K},
                    xlabel='t', ylabel='Population', title='Ms/K')

            else:  # n_obs == 4
                plot('EMF_hat_by_t', t_lst,
                    {'E_hat/K': y_hat_lst[:, 0]/K, 'M_hat/K': y_hat_lst[:, 1]/K, 'F_hat/K': y_hat_lst[:, 2]/K},
                    xlabel='t', ylabel='Population estimée', title='E_hat/K, M_hat/K, F_hat/K')
                plot('errors_by_t', t_lst,
                    {'(E-E_hat)/K': (y_lst[:, 0] - y_hat_lst[:, 0])/K,
                     '(M-M_hat)/K': (y_lst[:, 1] - y_hat_lst[:, 1])/K,
                     '(F-F_hat)/K': (y_lst[:, 2] - y_hat_lst[:, 2])/K},
                    xlabel='t', ylabel='Erreur', title='Erreurs d\'observation (E, M, F)')
                plot('Ms_hat_by_t', t_lst,
                    {'Ms/K': y_lst[:, 3]/K, 'Ms_hat/K': y_hat_lst[:, 3]/K},
                    xlabel='t', ylabel='Population', title='Ms/K, Ms_hat/K')
                plot('Ms_error_by_t', t_lst,
                    {'(Ms-Ms_hat)/K': (y_lst[:, 3] - y_hat_lst[:, 3])/K},
                    xlabel='t', ylabel='Erreur', title='Erreur d\'observation (Ms-Ms_hat)')

            # ── PLOT CONTRÔLE ─────────────────────────────────────────
            plot('u_by_t', t_lst, {'u': y_lst[:, 4]},
                xlabel='t', ylabel='u(t)', title='Contrôle u(t)', legend=False)

            # ── PLOT BREAKDOWN RÉCOMPENSES ────────────────────────────
            if reward_infos:
                # Trim in case 
                # n_min = min(len(v) for v in reward_infos.values())
                # reward_infos_trimmed = {k: v[:n_min] for k, v in reward_infos.items()}
                # plot('rewards_breakdown_by_t', env_times[0:n_min + 1],
                #     reward_infos_trimmed,
                #     xlabel='t', ylabel='r_i(t)',
                #     title='Breakdown of reward components by time')
                plot('rewards_breakdown_by_t', env_times,
                    reward_infos,
                    xlabel='t', ylabel='r_i(t)',
                    title='Breakdown of reward components by time')
                
            # Reward globale r_t
            plot('r_by_t', env_times, {'r': rewards},
                xlabel='t', ylabel='r(t)', title='Reward by time', legend=False)

            # Reward actualisée gamma^t * r_t
            plot('discounted_r_by_t', env_times, {'gamma^t * r_t': discounted_rewards},
                xlabel='t', ylabel='gamma^t * r(t)',
                title='Discounted reward by time', legend=False)
                
            # ── DUMP all recorded values at the current timestep ──
            self.logger.dump(self.num_timesteps)

        except Exception as e:
            import traceback
            print(f"\n[TensorboardCallback] CRASH à step {self.num_timesteps}")
            traceback.print_exc()