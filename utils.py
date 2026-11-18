from datetime import datetime
from pathlib import Path
from argparse import Namespace


from models import *


import os
def create_log_dir(subfolder):
    '''Creates a timestamped log directory under logs/<subfolder>/ and returns its Path.
    The process ID (PID) is added to ensure uniqueness of the directory name.
    Same as Nail's create_log_dir but for the LPSM cluster.'''
    now = datetime.now().strftime("%d%b%y_%Hh%Mm%Ss")
    timestamp = datetime.now().timestamp()
    # PID added
    pid = os.getpid()

    log_dir = Path(f"logs/{subfolder}/{int(timestamp)}_pid{pid}_{now}/")
    log_dir.mkdir(parents=True, exist_ok=False)

    return log_dir

def parse_sim_args(args):
    '''Defines the simulation class from args.sim and builds its kwargs from CLI arguments.
    Raises ValueError if the simulation name is unknown, TypeError if kwargs are invalid.'''

    available_sims = dict(filter(lambda kv: kv[0].startswith('Sim'), globals().items()))
    try:
        sim_class = available_sims[args.sim]
    except:
        raise ValueError(f'Simulation "{args.sim}" does not exist. Available simulations are: {", ".join(available_sims.keys())}.')

    # create sim
    try:
        print("DEBUG:", args)
        sim_params = {}
        # if args.u_types is not None: sim_params['u_types'] = eval(args.u_types) 
        if getattr(args, "u_types", None) is not None:
            sim_params["u_types"] = eval(args.u_types)
        if args.y0_without_u0 is not None: sim_params['y0_without_u0'] = args.y0_without_u0 
        if args.dt is not None: sim_params['dt'] = args.dt 
        sim_kwargs = {
            **sim_params,
            **eval(args.sim_kwargs)
        }
    except TypeError as err:
        raise TypeError(f'{args.sim}.{err}. Consider adding or removing some arguments using --kwargs or --sim_kwargs.')

    # return sim
    return sim_class, sim_kwargs

def parse_env_args(args):
    '''Builds the env_kwargs dict expected by RL_env from CLI args or a config dict.
    Accepts either an argparse Namespace or a plain dict (e.g. loaded from configs.json).'''
    if type(args) is dict:
        args = Namespace(**args)
    sim_class, sim_kwargs = parse_sim_args(args)

    env_kwargs = {
        'sim_class': sim_class,
        'sim_kwargs': sim_kwargs,
        'config': {
            'tmax': args.tmax,
            'action_min': args.action_min,
            'action_max': args.action_max,
            'n_past_states': args.n_past_states,
            'n_steps_per_action': args.n_steps_per_action,
            'discrete': args.discrete,
            'discrete_n_actions': args.discrete_n_actions
        }
    }
    
    return env_kwargs
