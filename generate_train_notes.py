"""
generate_exp_notes.py

Lit le configs.json d'un entraînement et génère un fichier de notes pré-rempli.

Usage:
    python generate_exp_notes.py logs/train/1780515356_03Jun26_21h35m56s/
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def generate_notes(exp_dir):
    exp_dir = Path(exp_dir)
    config_path = exp_dir / "configs.json"

    if not config_path.exists():
        raise FileNotFoundError(f"configs.json introuvable dans {exp_dir}")

    with open(config_path, "r") as f:
        configs = json.load(f)

    args = configs["args"]
    sim_kwargs_str = args.get("sim_kwargs", "dict()")
    # print("DEBUG: ", sim_kwargs_str)
    sim_kwargs = eval(sim_kwargs_str)  # récupère les kwargs comme dict

    # --- Calculs ---
    steps               = int(args["steps"])
    n_steps             = args["n_steps"]
    n_steps_per_action  = args["n_steps_per_action"]
    dt                  = args["dt"] if args["dt"] is not None else "?"
    tmax                = args["tmax"]
    cpus                = args["cpus"]

    if dt != "?":
        n_rollouts          = steps // (n_steps*cpus) 
        n_episodes          = round(steps * n_steps_per_action * dt / tmax, 1)
        # ep_per_rollout      = round(n_steps * n_steps_per_action * dt / tmax, 2)
        action_freq_days    = round(n_steps_per_action * dt, 4)
        mem_n               = args["mem_n_past_states"]
        mem_every           = args["mem_save_every"]
        history_days        = round(mem_n * mem_every * n_steps_per_action * dt, 2)
        # factor_K            = sim_kwargs["factor_K"]
    else:
        n_rollouts = ep_per_rollout = n_episodes = action_freq_days = history_days = "?"

    # --- Observables actifs ---
    obs_active = [k for k, v in sim_kwargs.items() if k.startswith("obs_") and v is True]
    ovitrap_period = sim_kwargs.get("ovitrap_period", 7.0)

    # --- Rewards ---
    rwd_y123        = sim_kwargs.get("rwd_y123", 0)
    rwd_y4          = sim_kwargs.get("rwd_y4", 0)
    rwd_y4_last100  = sim_kwargs.get("rwd_y4_last100", 0)
    gamma           = args["gamma"]

    # --- Action ---
    action_min  = args["action_min"]
    action_max  = args["action_max"]

    # --- Template ---
    template = f"""# Entraînement d'ID : {exp_dir.name}

## Épisodes et rollout
- Rollouts totaux     : {n_rollouts}   (= steps /(n_steps * cpus) = {steps} / {n_steps*cpus})
- Nb épisodes         : {n_episodes}   (= steps * n_steps_per_action * dt / tmax = {steps} * {n_steps_per_action} * {dt} / {tmax})
- Épisodes/rollout    : {n_episodes/n_rollouts}   
- CPUs (envs)         : {cpus}

## Récompenses
- rwd_y123       : {rwd_y123}
- rwd_y4         : {rwd_y4}
- rwd_y4_last100 : {rwd_y4_last100}
- gamma          : {gamma}

## Simulation
- dt             : {dt}
- tmax           : {tmax}


## Observables
- obs activés         : {', '.join(obs_active) if obs_active else 'aucun'}
- ovitrap_period      : (À VERIFIER SI = 7) {ovitrap_period} jours  (fréquence de mesure obs_E)
- historique (jours)  : {history_days}   (= mem_n_past_states * mem_save_every * n_steps_per_action * dt)
- mem_n_past_states   : {args['mem_n_past_states']}
- mem_save_every      : {args['mem_save_every']}

## Action
- n_steps_per_action  : {n_steps_per_action}
- période (jours)     : {action_freq_days}   (= n_steps_per_action * dt)
- action_min          : {action_min}
- action_max          : {action_max}

## Réseau
- hidden_layer_size   : {args['hidden_layer_size']}
- network_depth       : {args['network_depth']}
- lr                  : {args['lr']}
- n_epochs            : {args['n_epochs']}
- batch_size          : {args['batch_size']}

## Notes
-
"""

    # --- Sauvegarde ---
    output_path = exp_dir / "notes.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"Notes générées : {output_path}")
    # print(template)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str, help="Chemin vers le dossier d'expérience contenant configs.json")
    args = parser.parse_args()
    generate_notes(args.exp_dir)