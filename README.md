# RL_observer

Applying reinforcement learning to construct an observer for an ODE system.

## Installation

```
conda env create -f environment.yml
conda activate waves
```

## Differences with main 

- We train a model to construct an observer, not a control. 
- Models are trained with CPUs from the LPSM lab, and not Jean Zay as it is the case for main or Nail branch. The CPUs cluster from LPSM lab do not have slurm (cluster job manager) installed, so the command lines to launch trainings are probably different from main or Nail branch. Some of them are written in the `training_commands` folder. 