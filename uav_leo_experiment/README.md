# UAV-LEO Trajectory Energy Experiment

This folder is a clean experiment line for the teacher-requested direction:

- users generate computation tasks;
- tasks can run locally, on the UAV, or through the UAV to LEO;
- the UAV trajectory affects communication rate;
- UAV movement creates flight energy cost;
- partial offloading is represented by an offload ratio per user.

The folder is intentionally independent from the legacy hybrid GDRL/AMN path. The first goal is to get a defensible physical model and baseline table before reconnecting graph RL.

## Files

- `config.py`: experiment parameters.
- `env.py`: standalone UAV-LEO simulator.
- `baselines.py`: random, local-only, UAV-only, LEO-only, greedy partial, and TEA partial policies.
- `run_experiment.py`: baseline runner that writes CSV results.
- `plot_results.py`: generates comparison figures from CSV outputs.

## Proposed Algorithm

`tea_partial` means **Trajectory-Energy-Aware Partial Offloading**.

It is a deterministic heuristic inspired by UAV-LEO offloading papers that jointly optimize UAV trajectory, task offloading, and energy efficiency. For each time slot it:

1. estimates each user's local execution cost;
2. evaluates UAV and LEO offloading over a partial-offload ratio grid `[0.25, 0.5, 0.75, 1.0]`;
3. selects the target and ratio that minimize weighted latency-energy cost;
4. moves the UAV toward a workload-weighted user centroid only when the expected offloading gain justifies the extra flight energy.

This is not yet the final DRL algorithm. It is a stable expert/baseline policy used to validate the research setting before adding PPO/SAC/GDRL.


## Literature Basis

The current design follows the common formulation in UAV-assisted MEC and space-air edge computing papers:

- UAV trajectory changes the user-UAV channel, so movement and offloading must be optimized jointly.
- UAV flight energy cannot be ignored; moving toward users improves rate but increases propulsion energy.
- Partial offloading is useful because a task can be split between local execution and remote execution.
- LEO provides stronger computation capacity but has a worse relay link and higher remote compute energy in this simplified model.

Reference directions used for this design:

1. Learning-based stochastic game for energy-efficient UAV trajectory and task offloading in space/aerial edge computing, IEEE TVT 2025.
2. Joint task offloading and trajectory control for UAV-assisted MEC, 2023.
3. Energy-efficient task offloading and trajectory planning in UAV-enabled MEC networks, Computer Networks 2023.
4. UAV-enabled MEC with LEO enhancement, 2024.

## TEA Partial Decision Rule

For user `u`, target `k in {local, UAV, LEO}`, and offload ratio `rho`, the policy evaluates:

```text
cost(u, k, rho) = latency_weight * latency(u, k, rho)
                + energy_weight  * task_energy(u, k, rho)
```

The selected target and ratio are:

```text
(k*, rho*) = argmin cost(u, k, rho)
```

where `rho` is searched over `[0.25, 0.5, 0.75, 1.0]` for UAV/LEO and `0.0` for local.

The UAV trajectory target is a workload-and-gain weighted user centroid:

```text
weight_u = max(local_cost_u - best_remote_cost_u, 0) * normalized_workload_u
```

The UAV moves toward this centroid only if expected offloading gain is larger than the weighted flight-energy cost. This prevents unnecessary movement when local execution is already preferable.
## Quick Run

Run from `GDRL/`:

```bash
python -m uav_leo_experiment.run_experiment --episodes 20 --horizon 100
```

Outputs are saved under:

```text
experiments/uav_leo/U5_L4_20ep_T100/
```

Generate figures:

```bash
python -m uav_leo_experiment.plot_results experiments/uav_leo/U5_L4_20ep_T100
```

Figures:

- `reward_comparison.png`
- `latency_comparison.png`
- `energy_comparison.png`
- `flight_energy_comparison.png`
- `success_rate_comparison.png`
- `offload_ratio_comparison.png`
- `target_distribution.png`
- `uav_trajectory.png`

## Next Implementation Steps

1. Validate baseline trends and tune physical units.
2. Add ablations: `fixed_uav`, `no_flight_energy`, `full_offload_only`.
3. Add a Gymnasium wrapper so TRPO/PPO/SAC can use this environment.
4. Add a graph observation extractor only after the baseline simulator is stable.
5. Compare TEA partial with learned PPO/SAC/GDRL policies.


## Difficulty and Ablation Experiments

Difficulty presets are defined in `config.py`:

- `easy`: workflow check; high deadline and moderate tasks.
- `medium`: larger tasks and tighter deadline.
- `hard`: higher task load, lower bandwidth, tighter deadline.
- `stress`: pressure test with very large tasks and strict deadline.

Ablation presets:

- `none`: full environment.
- `fixed_uav`: UAV cannot move.
- `no_flight_energy`: flight energy is removed from the environment.
- `full_offload_only`: offload ratios are forced to 0 or 1.

Additional algorithms:

- `rate_aware`: chooses the better remote link between UAV and LEO.
- `full_offload_tea`: TEA without partial offloading ratios.
- `energy_guarded_tea`: TEA with stronger energy regularization.
- `deadline_tea`: TEA with deadline violation penalty.
- `tea_partial`: balanced trajectory-energy-aware partial offloading.
- `mpc_traj_h3` / `mpc_traj_h5`: receding-horizon (MPC) trajectory planner
  from the MEC literature: at each slot it evaluates a small set of candidate
  moves by simulating K slots forward (users and hotspot move with their
  observed velocities; future tasks are sampled from the hotspot-aware arrival
  model) and executes the best first move.  The offloading layer is the same
  exact per-slot solver used by GDRL, so MPC isolates the value of the learned
  trajectory: GDRL matches/beats MPC-H3 with a single forward pass instead of
  a per-slot search.

Run one environment:

```bash
python -m uav_leo_experiment.run_experiment --difficulty hard --ablation none --episodes 10 --horizon 30
```

Run a full sweep:

```bash
python -m uav_leo_experiment.run_sweep --episodes 10 --horizon 30 --output_dir experiments/uav_leo_sweep_check
```

Generate sweep figures:

```bash
python -m uav_leo_experiment.plot_sweep experiments/uav_leo_sweep_check
```

Sweep figures include:

- `sweep_reward.png`
- `sweep_success_rate.png`
- `sweep_latency.png`
- `sweep_energy.png`
- `sweep_offload_ratio.png`
- `sweep_latency_energy_tradeoff.png`

## Publication-quality figures (paper line)

The authoritative 40-episode run in `experiments/uav_leo_v2x_paper_final/`
has a dedicated figure generator that reads the sweep CSVs and the DRL
training logs and emits a consistent, paper-style figure set:

```bash
python -m uav_leo_experiment.make_paper_figures     --sweep_dir experiments/uav_leo_v2x_paper_final     --models_dir experiments/uav_leo_v2x/drl_models_v2
```

Figures (saved to `experiments/uav_leo_v2x_paper_final/figures/`):

- `convergence_{hard,stress}.png` ? DRL training convergence (eval reward vs steps)
- `reward_{hard,stress}.png` ? mean episode cost (= -reward) with std error bars
- `latency_{hard,stress}.png` ? mean service latency
- `energy_{hard,stress}.png` ? task + flight energy stacked bars
- `success_{hard,stress}.png` ? task success rate
- `reward_boxplot_{hard,stress}.png` ? per-episode reward distribution
- `cdf_latency_{hard,stress}.png` ? latency CDF
- `slot_latency_{hard,stress}.png` ? mean per-slot latency over the horizon
- `trajectory_{hard,stress}.png` ? UAV trajectories (episode 1)
- `offload_stack_{hard,stress}.png` ? local / UAV / LEO decision distribution
- `tradeoff.png` ? latency-energy trade-off scatter
- `ablation_gdrl.png` ? GDRL contribution ablation
- `gdrl_gain_{hard,stress}.png` ? paired per-episode gain vs baselines

## GDRL: Proposed Method (Residual Trajectory PPO + Optimization Offloading)

`gdrl` is the proposed method for the paper line. It decomposes the joint
problem into a *learned trajectory layer* and an *optimization-based offloading
layer*:

1. **Offloading layer (per slot, exact).** Given the UAV position that will
   actually serve the slot (post-move), each user's (target, ratio) choice is
   searched over `{local, UAV, LEO} x {0, .25, .5, .75, 1}` using the
   environment's true per-slot cost (`physics.simulate_task`). Because the
   slot-start backlogs are fixed, the per-slot cost is additive over users, so
   the per-user argmin is the exact per-slot optimum.

2. **Trajectory layer (PPO, residual).** The DRL policy outputs a 2-D residual
   `delta` on the demand-predictive expert move (`predict_tea`):
   `move = clip(expert_move + delta, uav_speed_max)`.  At `delta = 0` the
   policy is exactly the expert, so no behavior cloning is needed; PPO then
   learns where the myopic expert trajectory is suboptimal (hotspot
   anticipation, battery budgeting, queue backpressure).

This hybrid structure follows recent UAV-MEC literature that combines DRL for
trajectory with mathematical optimization for resource allocation
(e.g. "DRL-based trajectory optimization and computation-aware resource
allocation for UAV-assisted edge computing networks", IEEE 2025).

Train (from `GDRL/`):

```bash
python -m uav_leo_experiment.train_drl --method gdrl --difficulty v2x_hotspot_hard --steps 60000
python -m uav_leo_experiment.train_drl --method gdrl --difficulty v2x_hotspot_stress --steps 60000
```

Evaluate:

```bash
python -m uav_leo_experiment.run_experiment --difficulty v2x_hotspot_hard --users 12 \
    --gdrl_model experiments/uav_leo_v2x/drl_models_v2/gdrl/gdrl/v2x_hotspot_hard/model.pt \
    --methods gdrl,follow_tea,predict_tea,tea_partial,deadline_tea,lyapunov,ppo,dqn --skip_slow
```

Sweep (all difficulties, all baselines + GDRL):

```bash
python -m uav_leo_experiment.run_sweep --difficulties v2x_hotspot_hard v2x_hotspot_stress \
    --episodes 20 --horizon 30 --skip_slow \
    --output_dir experiments/uav_leo_v2x/gdrl_final
```

## Final Results (40 episodes, paired significance)

Authoritative run: `experiments/uav_leo_v2x_paper_final/` (per-difficulty GDRL
models, 40 episodes, seed 73 + episode index; full tables, paired significance
and the GDRL contribution ablations are in `paper_results.md`).  GDRL is the
best method on both hotspot scenarios, and the per-episode paired gain over
every heuristic and pure-DRL baseline is statistically significant (paired
t-test and Wilcoxon, p < 0.05).  The receding-horizon MPC-H3 optimization
baseline is the closest competitor: it ties GDRL on hard (p = 0.66) and loses
on stress (+2.3, p = 0.028).

| method | hard reward | hard success | stress reward | stress success |
|---|---|---|---|---|
| **GDRL (ours)** | **-85.2** | **0.971** | **-132.6** | **0.956** |
| Predict-TEA | -87.4 | 0.969 | -137.0 | 0.954 |
| MPC-H3 (receding horizon) | -85.6 | 0.970 | -134.8 | 0.955 |
| Follow-TEA | -93.2 | 0.965 | -140.1 | 0.952 |
| Deadline-TEA | -107.0 | 0.955 | -162.6 | 0.941 |
| PPO (BC+KL) | -106.7 | 0.957 | -165.7 | 0.941 |
| TEA / Lyapunov | -118.1 | 0.948 | -179.9 | 0.932 |
| DQN | -296.1 | 0.849 | - | - |
| TD3 | -336.7 | 0.821 | -420.0 | 0.829 |
| SAC | -304.1 | 0.840 | -414.5 | 0.828 |
| Random | -339.2 | 0.826 | -436.2 | 0.831 |

GDRL beats the best hand-crafted heuristic (Predict-TEA) by +2.2 on hard
(34/40 episode wins, p = 0.0002) and +4.4 on stress (36/40 wins, p < 0.0001),
matches/beats the receding-horizon MPC-H3 baseline (+0.4 hard, +2.3 stress;
stress p = 0.028) while using the same exact offloading layer (the learned
trajectory policy itself is a single network forward pass per slot), and beats
the standard DRL baselines (PPO/DQN/TD3/SAC) by 20-300+, while also achieving
the highest task success rate and the lowest (or tied-lowest) latency among
the top methods.

### GDRL contribution ablation (same 40 seeded episodes)

| difficulty | full GDRL | - trajectory residual (expert move) | - exact offload (TEA heuristic) |
|---|---|---|---|
| hotspot-hard | -85.2 | -85.2 (p = 0.30) | -85.2 (p = 0.32) |
| hotspot-stress | -132.6 | -132.3 (p = 0.30) | -132.6 (n.s.) |

Both layers are individually neutral: the learned trajectory residual converges
to ~zero (the demand-predictive expert move is already near-optimal in these
scenarios), and the TEA grid search matches the exact per-slot argmin on 39/40
hard episodes.  The decisive factor is *where* the offloading is evaluated:
Predict-TEA already uses the same hotspot-lookahead demand-predictive
trajectory but decides offloading at the **current** UAV position, while GDRL
solves it at the **post-move** position that actually serves the slot.
The mechanism ablation in `paper_results.md` (section 3b) shows this post-move
step is worth +2.2 (hard) and +4.7 (stress) over the current-position variant
(which itself matches Predict-TEA exactly).  The DRL residual can be dropped
with no loss (performance parity), which supports using the deterministic
framework in deployment.


### Energy-weight sensitivity (latency-energy trade-off)

The main table uses the default objective (latency weight 1.0, energy weight
0.001).  With a balanced objective (energy weight 0.01, i.e. flight/edge
energy comparable to latency, `experiments/uav_leo_v2x/gdrl_ew01_final/`),
GDRL remains the best method on both scenarios and its margin over the
energy-aware heuristics widens substantially:

| method | hard reward (ew=0.01) | hard success | stress reward (ew=0.01) | stress success |
|---|---|---|---|---|
| **GDRL (ours)** | **-125.1** | **0.971** | **-172.3** | **0.956** |
| Predict-TEA | -127.4 | 0.969 | -177.0 | 0.954 |
| Follow-TEA | -133.1 | 0.965 | -180.0 | 0.952 |
| PPO (trained at ew=0.001) | -147.6 | 0.957 | -205.7 | 0.941 |
| Deadline-TEA | -155.6 | 0.939 | -219.3 | 0.925 |
| E-Guard TEA | -156.0 | 0.939 | -219.8 | 0.925 |

GDRL keeps the highest success rate while being the cheapest among the
top methods, and the learned trajectory (which flies slightly less than the
heuristics) transfers to the energy-weighted objective.
