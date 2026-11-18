"""training.py"""

"""
Typical training command for the observer model (dated 23/07/2026), example with a
one output action (E_hat) and 4 outputs action (E_hat, M_hat, F_hat, Ms_hat):

  python training.py SimODE_v2 `
  --sim_kwargs "dict(rwd_y1=1.0, action_only_E=True, obs_E_state=True, obs_multi_scaling=True,factor_K=1)" `
  --u_types "{'resurgence_lin_deterministic': 1.0}" `
  --tmax 100.0 --dt 1e-2 `
  --gamma 0.99 `
  --action_min "np.array([0])" `
  --action_max "np.array([2*self.sim.K])" `
  --n_steps_per_action 10 `
  --batch_size 2048 --n_steps 2048 `
  --hidden_layer_size 256 --network_depth 2 `
  --cpus 10 --steps 30000 --checkpoint_interval 1500 --nb_eval_episodes 50 `
  --eval_callback False


python training.py SimODE_old `
  --sim_kwargs "dict(rwd_y123=0.1, rwd_y4=0.001, obs_y=True, obs_u=False, obs_E_state=False, obs_Ms=False, factor_K=1, dynamics_mode=True, dynamics_z=True, lambda_z=0.1, mu_z=0.1, min_value_obs=0)" `
  --u_types "{'u=kE': 0.25, 'u=kF': 0.25, 'resurgence_lin': 0.50}" `
  --tmax 1000.0 --dt 1e-2 `
  --action_min "np.array([-2*self.sim.K,-2*self.sim.K,-5*self.sim.K, -100 * self.sim.K])" `
  --action_max "np.array([2*self.sim.K,2*self.sim.K,5*self.sim.K, 100 * self.sim.K])" `
  --gamma 0.995 `
  --n_steps_per_action 100 `
  --y_period 1  --length_obs_history 1`
  --batch_size 2048 --n_steps 2048 `
  --hidden_layer_size 256 --network_depth 2 `
  --cpus 16 --steps 40000 --checkpoint_interval 1000 --nb_eval_episodes 50 `
  """

import argparse
# from datetime import datetime
from pathlib import Path
import numpy as np
import json
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.monitor import Monitor

from callbacks import TensorboardCallback, ProgressCallback

from env import RL_env
from utils import create_log_dir, parse_env_args



## Usefull function

def str2bool(v):
    '''Converts a string to a boolean value. Used for parsing CLI arguments such as eval_callback.'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Valeur booléenne attendue, reçu "{v}".')


# ---------------------------------------------------------------------------
#  Parsing of CLI parameters
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    'sim', type=str, help='Simulation to use, eg. "SimStabilizeMidObs".')


## Simulation parameters 
# sim params: all the input here have no default values in Sim_ODE or RL_env classes so that the class default
#  value do not get erased by the CLI one
parser.add_argument('--sim_kwargs', type=str, default='dict()',
                    help='Additional simulation kwargs, eg. "dict(f=lambda x: x*x)".')

parser.add_argument('--u_types', type=str, default = None,
                    help='List of type of control used in training, E.g \"[\"u=kE\", \"u=kF\"]\".')
parser.add_argument('--dt', type=float, default=None,
                    help='Time sampling interval. By default, use the simulation\'s default.')
parser.add_argument('--tmax', type=float, default=None,
                    help='Duration (in time) of one episode.')
parser.add_argument('--y0', type=str, default=None, help='Initial condition (E, M, F)(t=0)'
                    'By default, uses the simulation\'s default, if there is one.')
parser.add_argument('--y0_without_u0', type=str, default=None, help='Initial condition (E, M, F, Ms)(t=0) without u '
                    'By default, uses the simulation\'s default, if there is one.')


## RL environnement (RL_env, cf env.py) params 
parser.add_argument('--action_min', type=str, required=True, help='Minimum value of the observer. Could use a simulation-specific '
                    'constant, eg. -1.0 everywhere : \"np.array([-1.0, -1.0, -1.0, -1.0])\". Warning: the shape must match the'
                     ' observer\'s one (1, 3 or 4 usually)')
parser.add_argument('--action_max', type=str, required=True, help='Maximum value of the observer. Could use a simulation-specific '
                    'constant, eg. 1.0 everywhere : \"np.array([1.0, 1.0, 1.0, 1.0])\".Warning: the shape must match the observer\'s ' \
                    'one (1, 3 or 4 usually)')
parser.add_argument('--n_steps_per_action', type=int, default=1, help='Number of simulation steps for each environment step '
                    '(ie. the same action is applied several times if this is set to a value large than 1).')
parser.add_argument(
    '--on_time_reward', type=str2bool, default=False,
     help='False/True, default False. If True, the reward is computed before the n_steps_per_action steps'
          '(suitable for non dynamical observations action(t) = y_hat(t)). If False, the reward is computed only at the end of'
          ' the n_steps_per_action steps (suitable for dynamical observations, used also in control with RL).')


## Reinforcement learning training parameter
# Required params
parser.add_argument('--gamma', type=float, required=True, help='Gamma factor, usually 0.99') 
parser.add_argument('--cpus', type=int, required=True,
                    help='Number of CPUs to use for training. Usually ')
parser.add_argument('--steps', type=float, required=True,
                    help='Number of timesteps to train for.') # default=1e8
# Fixed params
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
parser.add_argument('--n_steps', type=int, default=1024, help='The number of steps to run for each environment per update '
                    '(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel).')
parser.add_argument('--batch_size', type=int,
                    default=1024, help='Minibatch size.')
parser.add_argument('--n_epochs', type=int, default=5,
                    help='Number of SGD iterations per epoch.')
parser.add_argument('--checkpoint_interval', type=int, default=500000,
                    help='Every so many steps, a checkpoint of the model will be saved.')
# Network
parser.add_argument('--network_depth', type=int, default=2,
                    help='Number of hidden layers in the policy and value networks.')
parser.add_argument('--hidden_layer_size', type=int, default=256,
                    help='Number of cells per hidden layer in the policy and value networks.')
# evaluation callback parameters
# For each jobs, the total number of CPUs used is then cpus + n_eval_envs  = 16 + 4 = 20 most of the times. 
parser.add_argument('--eval_callback', type=str2bool, default=True,
                     help='False/True, default True. If set, evaluates the model during training to choose the best model. ' \
                     )
parser.add_argument('--nb_eval_episodes', required=True, type=int,
                     help='Number of episodes that are run to make evaluation during training. 100 is ideal,' \
                     'but 50 seems enough, especially for trainings without much control diversity.')
parser.add_argument('--eval_interval', default=2.0, type=float, 
                     help='default 2.0   percents (of the total number of steps). Interval between two evaluations.')
parser.add_argument('--n_eval_envs', type=int, default=4,
                    help='default 4. Number of parallel environments used for evaluation callback.')


## Verbose, checkpoint, tracking with wanddb (wandb tracking is not used)
parser.add_argument('--verbose', default=False, action='store_true',
                    help='If set, prints training status periodically.')
parser.add_argument('--track', action='store_true', default=False, help='If set, tracks the experiment on wandb.')
parser.add_argument('--wandb_group', type=str, default=None, help='WandB group to group experiments together.')


args = parser.parse_args()



# ---------------------------------------------------------------------------
#  Create config
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # create experiment dir
    env_kwargs = parse_env_args(args)
    exp_dir = create_log_dir(subfolder='train')
    print(f'> {exp_dir}')

    print(f"(DEBUG): eval_callback {args.eval_callback}")
    print(f"(DEBUG): on_time_reward {args.on_time_reward}")
    # save config
    with open(exp_dir / 'configs.json', 'w', encoding='utf-8') as f:
        config_save = dict(
            args=vars(args), env_kwargs=copy.deepcopy(env_kwargs))
        config_save['env_kwargs']['sim_class'] = str(
            config_save['env_kwargs']['sim_class'])
        json.dump(config_save, f, ensure_ascii=False, indent=4)

    # Create vectorised environment used in the PPO 
    vec_env = make_vec_env(
    RL_env,
    n_envs=args.cpus,
    env_kwargs=env_kwargs,
    vec_env_cls=SubprocVecEnv if args.cpus > 1 else DummyVecEnv 
    )

    # Vectorised Evaluation environment used to save the best model.   
    eval_interval = int(args.steps * args.eval_interval * 1e-2)  # évalue tous les 2% des pas totaux
    N_EVAL_ENVS = args.n_eval_envs  # nombre de CPUs utilisés pour l'évaluation
    bool_eval_callback = args.eval_callback # evaluation callback or not
    if bool_eval_callback:
        eval_env = make_vec_env(
            RL_env,
            n_envs=N_EVAL_ENVS,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv if N_EVAL_ENVS > 1 else DummyVecEnv,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(exp_dir / "best_model"),
            log_path=str(exp_dir / "eval_logs"),
            eval_freq=max(eval_interval // args.cpus, 1),
            n_eval_episodes=args.nb_eval_episodes,   # sera réparti automatiquement en ~args.nb_eval_episodes/N_EVAL_ENVS par sous-env
            deterministic=True,
            render=False,
        )
    
    # Tensorboard evaluation callback to plot metrics of the training
    tb_eval_env = RL_env(**env_kwargs) 


    # create model
    model = PPO(
        'MlpPolicy',
        vec_env, verbose=int(args.verbose),
        tensorboard_log=exp_dir / 'tensorboard',
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        policy_kwargs={
            'net_arch': {
                'pi': [args.hidden_layer_size] * args.network_depth,
                'vf': [args.hidden_layer_size] * args.network_depth,
            }
        },
        device='cpu',
    )

    if args.track:
        import wandb
        wandb.init(entity='nathanlct', project="moustiques", group=args.wandb_group, sync_tensorboard=True, config=args)

    # create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq= max(args.checkpoint_interval // args.cpus, 1),
        save_path=str(exp_dir / 'checkpoints'),
        name_prefix='model',
        verbose=2 if args.verbose else 0,
    )
    tensorboard_callback = TensorboardCallback(tb_eval_env)

    # Callback to print the progress line of the training and an estimation of its duration
    #Remark: the tqdm progress bar in PPO do not displayed during training on linux so we implemented our own Progress callback. 
    progress_callback = ProgressCallback( 
        total_timesteps=int(args.steps),
        run_id=exp_dir.name,
        every_seconds=5,
    )

    # train model
    bool_eval_callback = args.eval_callback
    if bool_eval_callback: # no eval_callback to make tests of small trainings more quickly
        model.learn(
            total_timesteps=int(args.steps),
            progress_bar=False,
            callback=[checkpoint_callback, tensorboard_callback, progress_callback, eval_callback])
    else:
        print("\n**** CAUTION: EVAL CALLBACK DISABLED ****\n")
        model.learn(
            total_timesteps=int(args.steps),
            progress_bar=False,
            callback=[checkpoint_callback, tensorboard_callback, progress_callback])
        
    # save model
    model_path = exp_dir / 'model'
    model.save(model_path)
    print(f'> {model_path}')