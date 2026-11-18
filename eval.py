'''Evaluation of the model by computing trajectories plots and convergence histograms with several CPUs
 if specified in the n_workers variable.

Typical command line: 
    "python eval.py logs/train_saved/big_trainings/G5/1782634159_pid24196_28Jun26_10h09m19s/best_model/best_model.zip --n_workers 5"
    
with 
- n_traj (=10 by default): number of trajectories plotted
- n_hist (=200 by default): number of occurences in the histogram
- n_workers: number of CPUs that works in parallel

Parallelism: episode collection is distributed across all available CPU cores using
multiprocessing.Pool. Each worker loads its own copy of the model and env to avoid
shared-state issues across processes. Plotting and file saving happen in the main process,
preserving the original directory structure and naming conventions exactly.
'''

import numpy as np
from stable_baselines3 import PPO
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
from env import RL_env
from utils import parse_env_args

import os
from copy import deepcopy
import multiprocessing as mp
from functools import partial

# Force "spawn" start method: each worker starts as a fresh Python process instead of
# forking the parent. Prevents workers from re-importing the module and to be stuck. 
mp.set_start_method("spawn", force=True)

# Global variables 
_worker_model = None # RL model (used in _worker_init function)
_worker_env_kwargs = None # arguments (used in _worker_init function)
MAX_STEPS = 1e10 # max steps to avoid infinite looping in the functions. Value defined in the main 

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "cp_path",
    type=str,
    help="Path to .zip checkpoint to load and use "
    "as a control. There must be a configs.json in the parent directory.",
)
parser.add_argument(
    "--n_traj",
    type=int,
    default=10,
    help="Number of trajectories to plot (default: 10)",
)
parser.add_argument(
    "--n_hist",
    type=int,
    default=200,
    help="Number of simulations for histograms (default: 100)",
)
parser.add_argument(
    "--n_workers",
    type=int,
    default=None,
    help="Number of parallel worker processes (default: all available CPUs)",
)

parser.add_argument(
    "--u_types",
    type=str,
    default=None,
    help="Probability distribution of used controls",
)

parser.add_argument(
    "--tmax",
    type=float,
    default=None,
    help='Duration (in time) of one episode for the evaluation.',
)


args = parser.parse_args()


# ---------------------------------------------------------------------------
# Module-level worker function (must be picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _worker_init(model_path_str, env_kwargs):
    """Called once per worker process at pool creation. Loads the model and stores
    shared configuration in globals, avoiding repeated expensive loads per episode."""
    global _worker_model, _worker_env_kwargs
    _worker_model = PPO.load(model_path_str)
    _worker_env_kwargs = env_kwargs

def _run_episode(sim_idx):
    """Runs a single episode using the worker-local model and env. Creates a fresh env
    instance per episode (lightweight) while reusing the pre-loaded model (expensive)."""
    _env = RL_env(**_worker_env_kwargs)
    result = _env.run_episode(_worker_model, deterministic=True, max_steps=MAX_STEPS, tmax=args.tmax)
    return {
        "t": result["t"], "Y": result["Y"], "Y_hat": result["Y_hat"],
        "rewards": result["rewards"], "t_rewards": result["t_rewards"],
    }

def _collect_episodes_parallel(n_sims, model_path_str, env_kwargs,
                                n_workers, desc="Simulations"):
    """Parallel computations. Distributes n_sims episodes across n_workers processes 
    using a persistent pool. Workers are initialized once (model loaded once per worker),
    then episodes are dispatched in chunks to amortize spawn overhead."""

    chunksize = max(1, n_sims // n_workers)  # Number of simulations per workers
    results = []  

    with mp.Pool(processes=n_workers,        # creates n_workers process 
                 initializer=_worker_init,    # Load the PPO model once per worker
                 initargs=(model_path_str, env_kwargs)) as pool:
        # runs the episodes in parallel
        for res in tqdm(pool.imap(_run_episode, range(n_sims), chunksize=chunksize),  
                        total=n_sims, desc=desc):  # progress bar
            results.append(res)  # Store the result

    return results  

# ---------------------------------------------------------------------------
# Auxiliary functions
# ---------------------------------------------------------------------------

def get_convergence_time(array_t, error_array, eps):
    """Renvoit le premier temps à partir duquel l'erreur reste sous eps, ou None."""
    condition = error_array < eps

    assert not np.isnan(error_array).any(), "error_array contient des NaN"
    assert (error_array >= 0).all(), "error_array contient des valeurs négatives"

    for i in range(len(condition) - 1, -1, -1):
        if not condition[i]: # la variable au temps array_t[i] est > eps
            if i + 1 < len(array_t):  # Le temps de convergence est au temps d'après (s'il existe)
                return array_t[i + 1]
            else:
                return None # La condition n'est jamais atteinte définitivement
    # Tous les points vérifient la condition : convergence dès le début
    return array_t[0]

def worst_null_observer(obs_array, state_array, min_value_obs):
    """Donne le pire cas où l'observateur s'annule (i.e vaut min_value_obs) :
      Correspond à la valeur max de l'état du système aux instants où 
      l'observateur donné par le RL sature en obs = min_value_obs ."""
    obs_array = np.asarray(obs_array)
    state_array = np.asarray(state_array)

    if obs_array.shape[0] == 0 or state_array.shape[0] == 0:
        return None
    assert obs_array.shape == state_array.shape, "obs_array et state_array doivent avoir la même longueur"
    assert not np.isnan(obs_array).any() and not np.isnan(state_array).any(), "NaN détecté dans obs_array ou state_array"

    eps = 1e-12
    mask = obs_array <= min_value_obs + eps
    return state_array[mask].max() if mask.any() else None

# ---------------------------------------------------------------------------
# Plotting functions — now receive pre-collected results, no simulation inside
# ---------------------------------------------------------------------------

def plot_trajectories(results, env, configs,
                      show_plot=True, save_plot=False, title_folder=None, title_plot=None):
    """
    Figure with observation errors, real dynamics, observers, reward and control.
    Receives pre-collected episode results (list of dicts from _collect_episodes_parallel).
    File-saving logic is identical to the original.
    """
    fig, axes = plt.subplots(5, 3, figsize=(12, 15))
    gamma = configs['args']['gamma']

    colors = [
    "#e41a1c",  # rouge
    "#377eb8",  # bleu
    "#4daf4a",  # vert
    "#984ea3",  # violet
    "#ff7f00",  # orange
    "#ffff33",  # jaune
    "#a65628",  # marron
    "#f781bf",  # rose
    "#17becf",  # cyan
    "#bcbd22",  # olive
    ]

    if len(results) > len(colors):
        raise ValueError(f"Too many results ({len(results)}) for available colors ({len(colors)}).")

    for k, res in enumerate(results):
        # Adaptation for different actions: filling missing columns of Y_hat with nan values
        # nb_act : shape of the observer (usually 1, 3 or 4 :  E_hat, (E_hat, M_hat, F_hat) or (E_hat, M_hat, F_hat, Ms_hat)
        N_t, nb_act = res["Y_hat"].shape 
        # print(f"(DEBUG) action number : {nb_act} ")
        assert nb_act <= 4, f"nb_act ({nb_act}) ne peut pas dépasser 4 (E, M, F, Ms)"
        p = 4 - nb_act 
        fill_val = np.nan
        new_cols = np.full((N_t, p), fill_val)
        # print(f"(DEBUG) shape new cols: {new_cols.shape}")
        # always the last observers that are not considered  (e.g Ms_hat when (E_hat, M_hat, F_hat) is observed)
        array_Y_hat = np.concatenate((res["Y_hat"], new_cols),axis=1) 
        # print(f"(DEBUG) shape array_y_hat: {array_Y_hat.shape}")

        color = colors[k]
        array_t     = res["t"]
        array_Y     = res["Y"][:, :-1]   # drop last columns (control)
        array_u     = res["Y"][:, -1]
        rewards     = res["rewards"]
        t_rewards   = res["t_rewards"]

        discounted_reward = gamma ** np.arange(len(rewards)) * np.array(rewards)

        for i in range(4):  # E, M, F, Ms
            axes[i, 0].plot(array_t, array_Y[:, i]     / env.sim.K, color=color)
            axes[i, 1].plot(array_t, array_Y_hat[:, i] / env.sim.K, color=color)
            axes[i, 2].plot(array_t, (array_Y[:, i] - array_Y_hat[:, i]) / env.sim.K, color=color)

        axes[4, 0].plot(array_t,   array_u, color=color)
        axes[4, 1].plot(t_rewards, rewards, color=color)
        axes[4, 2].plot(t_rewards, discounted_reward, color=color)

    col_labels = ['Variable/K', 'Observateur/K', 'Erreur/K := (Var - Obs)/K']
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=12, fontweight='bold')

    row_labels = ['E', 'M', 'F', 'Ms']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=15, fontweight='bold')

    last_row_titles = ['Contrôle u(t)', r'Récompense $r_t$', r'Récompense actualisée $\gamma^t r_t$']
    for j, title in enumerate(last_row_titles):
        axes[4, j].set_title(title, fontsize=9)
        axes[4, j].set_xlabel('Temps (jours)', fontsize=8)

    if title_plot is not None:
        fig.suptitle(title_plot, fontsize=12, y=0.98)
        fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.5, rect=[0, 0, 1, 0.96])

    if save_plot:
        # ---- unchanged from original ----
        save_dir = os.path.join("plot_eval_train", title_folder)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "trajectoires.png"))

    if show_plot:
        plt.show()

def plot_histograms(results, env, n_sims,
                    show_plot=True, save_plot=False, title_folder=None,
                    title_plot=None, experiment_ref=None, bool_test_null_observer=False):
    """
    Figure with histograms of final values and convergence times at 1% and 5%.
    Receives pre-collected episode results. File-saving logic is identical to the original. 
    Handles several shape of action (nb_act = 1, 3 or 4) automatically.
    """
    assert len(results) == n_sims, "n_sims ne correspond pas au nombre de résultats"

    nb_act = results[0]["Y_hat"].shape[1] # shape of the observer (E, (E, M, F), (E, M, F, Ms) usually)
    labels = ['E', 'M', 'F', 'Ms'][:nb_act]

    fig, axes = plt.subplots(nb_act, 3, figsize=(15, 3 * nb_act))
    if nb_act == 1:
        axes = axes[np.newaxis, :]

    final_lists = [[] for _ in range(nb_act)]
    convergence_lists_1 = [[] for _ in range(nb_act)]
    convergence_lists_5 = [[] for _ in range(nb_act)]
    null_lists = [[] for _ in range(nb_act)]

    for sim_idx, res in enumerate(results):
        array_t = res["t"]
        array_Y = res["Y"][:, :-1]
        array_Y_hat = res["Y_hat"]

        if bool_test_null_observer:
            for k in range(nb_act):
                worst = worst_null_observer(obs_array=array_Y_hat[:, k],
                                             state_array=array_Y[:, k], min_value_obs=env.sim.min_value_obs)
                if worst is not None:
                    null_lists[k].append(worst)

        for k in range(nb_act):
            error = np.abs(array_Y[:, k] - array_Y_hat[:, k]) / env.sim.K
            final_lists[k].append(error[-1])

            t_ext_1 = get_convergence_time(array_t, error, 0.01)
            t_ext_5 = get_convergence_time(array_t, error, 0.05)

            if t_ext_1 is not None:
                convergence_lists_1[k].append(t_ext_1)
            if t_ext_5 is not None:
                convergence_lists_5[k].append(t_ext_5)

    for k, label in enumerate(labels):
        axes[k, 0].hist(final_lists[k], bins='auto', edgecolor='black', color='mediumseagreen', alpha=0.85)
        axes[k, 0].set_xlabel(f'|{label} - {label}_hat| / K', fontsize=13)
        axes[k, 0].set_ylabel('Occurrence(s)', fontsize=13)

        data_1 = convergence_lists_1[k]
        axes[k, 1].hist(data_1, bins='auto', edgecolor='black', color='mediumseagreen', alpha=0.85)
        axes[k, 1].set_xlabel(f"{label}-{label}_hat : {len(data_1)/n_sims*100:.1f}% ({len(data_1)}/{n_sims} succès)", fontsize=13)
        axes[k, 1].set_ylabel('Occurrence(s)', fontsize=13)

        if len(data_1) > 0:
            mean_val = np.mean(data_1)
            axes[k, 1].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'moy = {mean_val:.1f}j')
            axes[k, 1].legend(fontsize=12)

        data_5 = convergence_lists_5[k]
        axes[k, 2].hist(data_5, bins='auto', edgecolor='black', color='mediumseagreen', alpha=0.85)
        axes[k, 2].set_xlabel(f"{label}-{label}_hat : {len(data_5)/n_sims*100:.1f}% ({len(data_5)}/{n_sims} succès)", fontsize=13)
        axes[k, 2].set_ylabel('Occurrence(s)', fontsize=13)

        if len(data_5) > 0:
            mean_val = np.mean(data_5)
            axes[k, 2].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'moy = {mean_val:.1f}j')
            axes[k, 2].legend(fontsize=12)

    col_titles = ["Valeur finale", "Temps de convergence (jours), 1% de K", "Temps de convergence (jours), 5% de K"]
    for j, col_title in enumerate(col_titles):
        axes[0, j].set_title(col_title, fontsize=13, fontweight='bold')

    if title_plot is not None:
        fig.suptitle(title_plot, fontsize=12, y=0.98)

    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.5, rect=[0, 0, 1, 0.96])

    if save_plot:
        save_dir = os.path.join("plot_eval_train", title_folder)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "histogrammes.png"))

        txt_path = os.path.join(save_dir, "mean_CV_time.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            ref = experiment_ref if experiment_ref is not None else title_folder
            f.write(f"Référence expérience : {ref}\n")
            f.write(f"Nombre d'essais      : {n_sims}\n")
            f.write(f"K = {env.sim.K}\n")

            labels_string = ", ".join(labels)

            for threshold_percent, convergence_lists in [(1, convergence_lists_1), (5, convergence_lists_5)]:
                f.write(f"\n--- Seuil {threshold_percent}% de K ---\n")
                f.write(f"Temps moyens de convergence (jours) pour ({labels_string}) : (")
                for k in range(nb_act):
                    f.write(f"{int(np.mean(convergence_lists[k])) if len(convergence_lists[k]) > 0 else 0}" + (", " if k < nb_act - 1 else ")"))
                f.write("\n")

                f.write(f"Taux de succes de convergence (%) pour ({labels_string}) : (")
                for k in range(nb_act):
                    f.write(f"{len(convergence_lists[k])/n_sims*100:.1f}" + (", " if k < nb_act - 1 else ")"))
                f.write("\n")

            mean_rewards = [np.mean(res["rewards"]) for res in results]
            f.write("\n--- Récompense moyenne ---\n")
            f.write(f"Récompense moyenne sur {n_sims} essais : {np.mean(mean_rewards):.2f}\n")

            if bool_test_null_observer:
                rates = [len(l) / n_sims * 100 for l in null_lists]
                means = [np.mean(l) / env.sim.K if len(l) > 0 else 0 for l in null_lists]
                stds = [np.std(l) / env.sim.K if len(l) > 0 else 0 for l in null_lists]

                f.write("\n--- Problème d'annulation d'observateur ---\n")
                f.write(f"Taux (en %): ({', '.join([f'{lab}={val:.1f}' for lab, val in zip(labels, rates)])})\n")
                f.write(f"État moyen (/K) : ({', '.join([f'{lab}={val:.2f}' for lab, val in zip(labels, means)])})\n")
                f.write(f"Écart type (/K) : ({', '.join([f'{lab}={val:.2f}' for lab, val in zip(labels, stds)])})\n")

    if show_plot:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Load model and config
    # ---------------------------------------------------------------------------
    model_path = Path(args.cp_path)
    if ("checkpoints" in str(model_path)) or ("best_model" in str(model_path)):
        with open(str(model_path.parent.parent / "configs.json"), "r") as f:
            configs = json.load(f)
    else:
        with open(str(model_path.parent / "configs.json"), "r") as f:
            configs = json.load(f)
    env_kwargs = parse_env_args(configs["args"])


    # ---------------------------------------------------------------------------
    # Load control choice and create env for reference. 
    #---------------------------------------------------------------------------
    if args.u_types is not None:
        parsed = eval(args.u_types)
        env_kwargs["sim_kwargs"]["u_types"] = parsed            
    else: # On évalue avec la même distribution de contrôle qui a servi à entraîner le modèle
        if env_kwargs["sim_kwargs"].get("u_types", None) is None:
            env_kwargs["sim_kwargs"]["u_types"] = {"u=kE": 1.0}  # rétrocompatibilité
        else:
            parsed = env_kwargs["sim_kwargs"]["u_types"]
            if isinstance(parsed, list):
                env_kwargs["sim_kwargs"]["u_types"] = {"u=kE": 0.5, "u=kF": 0.5} # rétropcompatibilité 
            else: 
                env_kwargs["sim_kwargs"]["u_types"] = parsed


    ctrl_dict = env_kwargs["sim_kwargs"]["u_types"]
    foldername_ctrl_type =  "_".join(f"{k}-{v}" for k, v in sorted(ctrl_dict.items()))
    # print(f"(DEBUG) ctrl file filename: {foldername_ctrl_type}")
    
    # Env creation
    env = RL_env(**env_kwargs)
    N_WORKERS = args.n_workers if args.n_workers is not None else mp.cpu_count()
    print(f"[parallel] Using {N_WORKERS} worker processes on {mp.cpu_count()} available CPUs.")

    n_traj   = args.n_traj
    n_hist   = args.n_hist
    # Indexes of the simulations where the observer vanishes while the state is still high.
    # Used to fix a null observer issue
    bool_test_null_observer = True 
    model_path_str = str(model_path)


    # ---------------------------------------------------------------------------
    # Preparing file titles 
    # ---------------------------------------------------------------------------
    run_folder = (
        model_path.parent.parent
        if ("checkpoints" in str(model_path)) or ("best_model" in str(model_path))
        else model_path.parent
    )
    folder_name = run_folder.name
    # print("DEBUG : ", folder_name)
    parts  = folder_name.split("_")
    run_id = parts[0] + "_" + parts[1]

    if "checkpoints" in str(model_path):
        cp_name    = model_path.stem
        n_steps_cp = cp_name.split("_")[1]
        total_steps = configs["args"].get("steps", "?")
        model_label = f"{run_id} -- checkpoint {n_steps_cp}/{total_steps} steps"
    elif "best_model" in str(model_path):
        data = np.load(model_path.parent.parent / "eval_logs" / "evaluations.npz")
        timesteps = data["timesteps"]
        results = data["results"]  # shape (n_evals, n_eval_episodes)
        mean_rewards = results.mean(axis=1)
        best_idx = mean_rewards.argmax()

        model_label = f"{run_id} -- meilleur modèle ( {timesteps[best_idx]} steps)"
    else:
        total_steps = configs["args"].get("steps", "?")
        model_label = f"{run_id} -- Modèle final ({total_steps} steps)"

    title_plot_traj = model_label + f" -- trajectoires sur {n_traj} épisodes"
    title_plot_hist = model_label + f" -- métriques sur {n_hist} épisodes"

    # Different tmax
    tmax_title = "T_default" if args.tmax is None else f"T_{args.tmax}"
    if "checkpoints" in str(model_path):
        title_folder = Path(folder_name + "_plot") / tmax_title / f"checkpoints_{n_steps_cp}"
    elif "best_model" in str(model_path):
        title_folder = Path(folder_name + "_plot") / tmax_title / "best_model"
    else:
        title_folder = Path(folder_name + "_plot") / tmax_title / "final_model"

    title_folder = title_folder / foldername_ctrl_type

    # print("DEBUG: ", title_folder)


    # ---------------------------------------------------------------------------
    #  Collect episodes in parallel 
    # ---------------------------------------------------------------------------
    # Trajectories and histograms share independent episodes; collect separately so each set has its own i.i.d. rollouts.
    print(f"\n[parallel] Collecting {n_traj} trajectory episodes...")
    traj_results = _collect_episodes_parallel(
        n_sims=n_traj,
        model_path_str=model_path_str,
        env_kwargs=env_kwargs,
        n_workers=N_WORKERS,
        desc="Trajectoires",
    )

    print(f"\n[parallel] Collecting {n_hist} histogram episodes...")
    hist_results = _collect_episodes_parallel(
        n_sims=n_hist,
        model_path_str=model_path_str,
        env_kwargs=env_kwargs,
        n_workers=N_WORKERS,
        desc="Histogrammes",
    )


    # ---------------------------------------------------------------------------
    #  Plot and save (main process only, identical filenames/dirs) 
    # ---------------------------------------------------------------------------
    plot_trajectories(
        traj_results, env=env, configs=configs,
        show_plot=False, save_plot=True,
        title_folder=title_folder, title_plot=title_plot_traj,
    )

    plot_histograms(
        hist_results, env=env, n_sims=n_hist,
        show_plot=False, save_plot=True,
        title_folder=title_folder, title_plot=title_plot_hist,
        experiment_ref=title_folder,
        bool_test_null_observer=bool_test_null_observer
    )