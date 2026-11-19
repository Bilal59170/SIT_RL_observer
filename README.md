# The main information
This repository contains the Python code, report and presentation of my internship at LPSM (Laboratoire de Probabilités, Statistique et Modélisation) Sorbonne laboratory during my final year at École Nationale des Ponts et Chaussées. This internship was supervised by [Luis Almeida](https://www.ljll.fr/~almeida/), [Jean-Michel Coron](https://www.ljll.fr/~coron/) and [Amaury Hayat](https://cermics.gitlab.enpc.fr/hayat/), in collaboration with [Nail Baloul](https://nailbaloul.github.io/). 

In this ``read_me.md`` file, we describe some of the important parts of this work, especially for observer construction. For more details, feel free to take a look at the report ``Rapport_Stage_PFE_Bilal_Ben-Moussa.pdf`` or a presentation of this work `Presentation_orale_PFE_Bilal_Ben-Moussa.pdf`. Both documents are written in French. 

Overview: 
- In this ``read_me.md``: quick description of the mathematical context and the code first. Some details about the context are added afterwards.
- `handouts` folder: contains the report (`Rapport_Stage_PFE_Bilal_Ben-Moussa.pdf`) and presentation (`Presentation_orale_PFE_Bilal_Ben-Moussa.pdf`) of this work.  
- Other folders and files are used for training and evaluation of reinforcement learning models. 


# Quick Mathematics
This GitHub repository contains code to construct an observer (or state estimator) of an ODE system with partial observation using reinforcement learning.

The EDO is a mosquito population model where only $E,$ the number of eggs, is observed:
```math
\begin{cases}
    \dot{E} = \beta_E F\left(1-\frac{E}{K}\right)\frac{M}{M+\gamma_s M_s}-(\nu_E+\delta_E)E, \\
    \dot{M} = (1-\nu)\nu_E E-\delta_M M, \\
    \dot{F} = \nu\nu_E E-\delta_F F, \\
    \dot{M_s} = u-\delta_s M_s, \\
    y=E.
\end{cases}
```
Denoting $x = (E, M, F, M_s)^T$, we rewrite the system as 
```math
\begin{cases}
    \dot{x} = f(x, u),\\
    y = Cx, \quad C = (1, 0, 0, 0).
\end{cases}
```
From observations of the dynamics of $y=E$ alone, the reinforcement learning algorithm computes an observer $\widehat{x}$ which is an approximation of the trajectory of the whole system: 
```math
\begin{equation*}
	\forall x(0),\widehat{x}(0),\quad e(t)\coloneqq\widehat{x}(t)-x(t)\xrightarrow[t\to+\infty]{}0.
\end{equation*}
```
We considered two ways of computing the observer by reinforcement learning. The first way is the direct approach where RL agent directly outputs the observer. Here $\mathcal{N}$ is a normalisation function and $a$ is the action of the agent:

```math
\begin{equation*}
	\widehat{x}(t+dt) = \mathcal{N}(a(t)) = \mathcal{N}(\pi_{\theta}(\widehat{x}(t), E(t), u(t))).
\end{equation*}
```

The second is a Luenberger-like observer where $\widehat{x}(\cdot)$ is computed from its dynamics with unknown gain matrix $L(t)$ which is determined by the RL agent: 
```math
\begin{cases}
	\frac{\widehat{x}(t+dt)-\widehat{x}(t)}{dt} = f_4(\widehat{x}(t), u(t)) + L(\widehat{E}(t) - E(t)), \\
	L(t) = a(t) = \pi_{\theta}(\widehat{x}(t), E(t), u(t)) \in [-1, 1]^4.
\end{cases}
```

For more details on these observers or the implementation of the RL algorithm, see the "Details on the internship" part below or part III of the report. 


# Code

**Differences with main branch.** We train a model to construct an observer, not a control.

## Installation

```
conda env create -f environment.yml
conda activate waves
```

## Training 
The training of the RL agent with the **PPO** algorithm in `Gymnasium` is configured in `training.py`. `callbacks.py` contains the callbacks used with tensorboard to visualise the training metrics. Typical training commands are given at the beginning of `training.py` on Windows or Linux. 

## Evaluation
We provided two RL models, one for a Luenberger-like model and the other using a direct method.
- 10 trajectories are computed for plots
- 200 other trajectories are computed for convergence evaluation, but are not plotted.
- Execution time is about 30s for the 10 trajectories and 8 min for the 200 trajectories. 

Here are the commands to evaluate them.

Luenberger: 
```bash
python eval.py logs/Luenberger/1786364461_pid32824_10Aug26_14h21m01s/model.zip --n_workers 2 --observer_mode "luenberger_partial"
```

Direct: 
```bash
python eval.py logs/Direct/1785937108_pid47588_05Aug26_15h38m28s/best_model/best_model.zip --n_workers 2
```

The evaluation will produce plots that will be added to the `plot_eval_train` folder. The ``checkpoints`` and ``tensorboard`` files for the direct approach model are too large to be commited, so they are empty in this repo. We also removed those two folders for the Luenberger approach model. For some details about the plots, see parts III.2 and III.3 of the report. 

# Details on the internship: Mathematics and Motivation

During this 5-month internship, we took an interest in a new system of equations to model mosquito population dynamics and tackled control and observation problems.  More precisely, we derived basic control properties of the new system of equations and constructed an **observer** using **reinforcement learning**.


  
## The equations and motivation

We considered the system of equations that models the dynamics of a mosquito population: 

```math
\begin{cases}
\dot{E} = \beta_E F\left(1-\frac{E}{K}\right)\frac{M}{M+\gamma_s M_s}-(\nu_E+\delta_E)E, \\
\dot{M} = (1-\nu)\nu_E E-\delta_M M, \\
\dot{F} = \nu\nu_E E-\delta_F F, \\
\dot{M_s} = u-\delta_s M_s.
\end{cases}
```

Here, $x=(E,M,F,M_s)^T$ denotes the population state: $E$ is the number of eggs, $M$ the number of wild adult males, $F$ the number of adult females considered as mated in the model, and $M_s$ the number of sterile adult males. The input $u(t)\geq 0$ is the release rate of sterile males.



The goal is to 
reduce the wild mosquito population using the **Sterile Insect Technique (SIT)**: sterile males are released into the environment to compete with wild males for mating. The factor $M/(M+\gamma_s M_s)$ models the proportion of fertile matings. We seek release strategies $u$ that drive $(E,M,F)$ toward zero while limiting the number of sterile males released.

As we want our study to be used in real-life settings, this model uses the population $F$ of mated females rather than only fertile females $F_f$, unlike previous models from ``bidi2025``, making the relevant female population **easier to measure in the field**. We also introduce **partial observations** at discrete times and try to construct an observer. This task is described in the "Observation" part.

## Mathematical results

### Control
We found that we had the same properties as the previous model of `bidi2025`: 

```math
\begin{cases}
\dot{E} = \beta_E F_f\left(1-\frac{E}{K}\right)- (\nu_E+\delta_E)E, \\
\dot{M} = (1-\nu)\nu_E E-\delta_M M, \\
\dot{F_f} = \nu\nu_E E\frac{M}{M+\gamma_s M_s}-\delta_F F_f, \\
\dot{M_s} = u-\delta_s M_s.
\end{cases}
```

Here, $F_f$ denotes females mated with fertile males. In this earlier model, the mating ratio acts on recruitment into $F_f$; in our model, it acts directly on egg production.

The extinction result concerns the wild population $(E,M,F)$. With a constant release rate $u=\overline{U} > U^*$ where $U^*$ is a specific value that we derived analytically, the wild population tends to zero. In practice, it means that we can erase the mosquito population with a finite-cost control. 

### Observation
An observer $\widehat{x}$ is an approximation of the trajectory of the system: 
```math
\begin{equation*}
	\forall x(0),\widehat{x}(0),\quad e(t)\coloneqq\widehat{x}(t)-x(t)\xrightarrow[t\to+\infty]{}0.
\end{equation*}
```

It is often constructed using the  dynamics of the system, e.g., Luenberger observers for linear control systems. In our case, we only observe the number of eggs $E.$ In the case of a nonlinear system like ours, `Observer_design_EDO` gives a similar structure under some conditions:  

```math
\begin{cases}
	\dot{x}&=f(x,u),\\
	y&=Cx=E, \quad C=\begin{pmatrix}1&0&0&0\end{pmatrix},\\
	\dot{\widehat{x}}&=f(\widehat{x},u)+L(C\widehat{x}-y).
\end{cases}
```

Here, $f$ is the nonlinear dynamics and $L\in\mathbb{R}^{4\times 1}$ is a correction gain. 

The nonlinear observer result mentioned in the presentation requires additional assumptions, including a bounded Jacobian of the dynamics. For our model, this boundedness is problematic near $(M,M_s)=(0,0)$ because of the mating ratio. We could not prove these assumptions. Instead, we used reinforcement learning to learn the weights $L$ to see whether this form of observer works. 
## Reinforcement learning (RL)
### Link between Math and RL 

In **reinforcement learning**, an agent interacts with a dynamical environment through partial observations, rewards, and actions. A policy $\pi_\theta$ maps the agent's observation $o_t$ to an action $a_t$. Training seeks to maximize the expected discounted cumulative reward:

```math
J(\pi_\theta)=\mathbb{E}_{\tau\sim(\pi_\theta,\mathcal{M})}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right],
```

where $\mathcal{M}$ denotes the environment, $\tau$ an interaction trajectory, and $\gamma\in[0,1)$ the discount factor. This objective is closely related to optimal control, but here we use it to learn a state estimator rather than a release strategy.

For observer construction, we use the same methodology as in `agbobidi_RL`. The environment evolves the mosquito dynamics and the estimated state using an Euler time discretization. Its state includes $(x,u,\widehat{x})$, while the agent receives only

```math
o_t=(\widehat{x}(t),E(t),u(t)).
```

During training, the simulated true state is used to evaluate the estimation error through the reward

```math
r_t=-c_1\frac{\left\|(\widehat{E}-E,\widehat{M}-M,\widehat{F}-F)(t)\right\|_2}{K}-c_2\frac{|(\widehat{M_s}-M_s)(t)|}{K},\qquad c_1,c_2>0.
```

Thus, maximizing reward encourages accurate state reconstruction without providing the unobserved true populations to the policy. We investigate two choices of action: learning the correction gain $L$ of a Luenberger-type observer, or directly predicting the next estimated state. Numerical experiments consider observation intervals of $1$, $3.5$, and $7$ days. 

### Luenberger structure
We computed the observer from random initial conditions and dynamics:

```math
\begin{cases}
	\frac{\widehat{x}(t+dt)-\widehat{x}(t)}{dt} = f_4(\widehat{x}(t), u(t)) + L(\widehat{E}(t) - E(t)), \\
	L(t) = a(t) = \pi_{\theta}(\widehat{x}(t), E(t), u(t)) \in [-1, 1]^4.
\end{cases}
```
This strategy turned out to work as $\widehat{x}(\cdot)$ converges numerically to $x(\cdot)$ in 10-20 time units (here days). To see the results, see e.g. slide 41/62 of the presentation or page 65/70 of the report. 

### Direct approximation 
The direct approximation consisted of giving the observer the agent's output directly. This task is more difficult a priori because we do not use any information from the dynamical system, except in the reward. Here $\mathcal{N}$ is a linear normalization function:  

```math
\begin{equation*}
	\widehat{x}(t+dt) = \mathcal{N}(a(t)) = \mathcal{N}(\pi_{\theta}(\widehat{x}(t), E(t), u(t))).
\end{equation*}
```

This strategy gives more mixed results with dynamics that look like the target dynamics but with large numerical discrepancies. To see the results, see e.g. slide 43/62 of the presentation or page 57/70 of the report. 

## Citations
- `bidi2025`:  Agbo Bidi, K., Almeida, L., & Coron, J.-M. (2025). Global stabilization of a sterile insect technique model by feedback laws. Journal of Optimization Theory and Applications, 204(2), 30. https://doi.org/10.1007/s10957-024-02566-4
- ``Observer_design_EDO``: Zemouche, A., Boutayeb, M., & Bara, G. I. (2005). Observer design for nonlinear systems: An approach based on the differential mean value theorem. In Proceedings of the 44th IEEE Conference on Decision and Control (pp. 6353–6358). https://doi.org/10.1109/CDC.2005.1583180
- ``agbobidi_RL``: Agbo Bidi, K., Coron, J.-M., Hayat, A., & Lichtlé, N. (2025). A novel approach to feedback control with deep reinforcement learning. Systems & Control Letters, 202, 106102. https://doi.org/10.1016/j.sysconle.2025.106102
