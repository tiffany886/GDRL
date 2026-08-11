# UAV-LEO Task Offloading & Trajectory Optimization — Paper Results

> Generated automatically from the 40-episode main sweep and the GDRL contribution ablations. Paired statistics use per-episode rewards of the same seeded episodes (seed 73 + episode index).

## 1. Evaluation scenarios

| parameter | hotspot-hard | hotspot-stress |
|---|---|---|
| users | 12 | 16 |
| horizon (slots) | 30 | 30 |
| user-UAV bandwidth | 1.0 MHz | 1.1 MHz |
| task deadline | 0.65 s | 0.60 s |
| user CPU | 5.0 GHz | 5.0 GHz |
| UAV edge CPU | 50.0 GHz | 60.0 GHz |
| LEO edge CPU | 1.0 THz | 1.0 THz |
| task size | 0.8-3.0 Mb, 900-1700 cyc/bit | 0.8-3.0 Mb, 1000-1900 cyc/bit |
| UAV-LEO backhaul | 20 Mbps | 20 Mbps |
| path-loss exponent | 3.0 | 3.0 |
| hotspot motion | radius 100 m, 18 m/s | radius 90 m, 20 m/s |
| drop penalty | 4.0 | 4.0 |
| energy weight | 0.001 | 0.001 |

## 2. Main results — v2x_hotspot_hard

| rank | method | reward | success | latency (s) | energy | offload | drop | Δ vs PMEO | wins | p (t / Wilcoxon) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GDRL (ours) | -85.2 | 0.971 | 0.1071 | 147.7 | 0.207 | 0.029 | -0.0 | 10/40 | 0.0930 / 0.0436 |
| 2 | PMEO (ours) | -85.2 | 0.971 | 0.1072 | 148.2 | 0.207 | 0.029 |  — | — | — |
| 3 | MPC-H3 (receding horizon) | -85.6 | 0.970 | 0.1074 | 124.0 | 0.207 | 0.030 | +0.4 | 14/40 | 0.6768 / 0.6179 |
| 4 | Current-pos exact | -87.4 | 0.969 | 0.1073 | 148.2 | 0.206 | 0.031 | +2.2 | 37/40 | 0.0002 / 0.0000 |
| 5 | Predict-TEA | -87.4 | 0.969 | 0.1073 | 148.2 | 0.206 | 0.031 | +2.2 | 37/40 | 0.0002 / 0.0000 |
| 6 | Follow-TEA | -93.2 | 0.965 | 0.1067 | 148.1 | 0.208 | 0.035 | +7.9 | 31/40 | 0.0000 / 0.0000 |
| 7 | MPC-H10 (receding horizon) | -93.9 | 0.964 | 0.1087 | 104.1 | 0.207 | 0.036 | +8.7 | 24/40 | 0.0010 / 0.0042 |
| 8 | PPO (BC+KL) | -106.7 | 0.957 | 0.1130 | 148.7 | 0.199 | 0.043 | +21.5 | 40/40 | 0.0000 / 0.0000 |
| 9 | Deadline-TEA | -107.0 | 0.955 | 0.1106 | 92.4 | 0.208 | 0.045 | +21.8 | 34/40 | 0.0000 / 0.0000 |
| 10 | Transformer-PPO (2025) | -117.7 | 0.950 | 0.1152 | 149.3 | 0.205 | 0.050 | +32.4 | 40/40 | 0.0000 / 0.0000 |
| 11 | TEA | -118.1 | 0.948 | 0.1117 | 87.5 | 0.209 | 0.052 | +32.9 | 36/40 | 0.0000 / 0.0000 |
| 12 | Lyapunov | -118.2 | 0.948 | 0.1117 | 87.5 | 0.209 | 0.052 | +33.0 | 36/40 | 0.0000 / 0.0000 |
| 13 | GAT-PPO (2024) | -120.0 | 0.949 | 0.1187 | 149.1 | 0.210 | 0.051 | +34.8 | 40/40 | 0.0000 / 0.0000 |
| 14 | E-Guard TEA | -120.0 | 0.947 | 0.1135 | 88.5 | 0.208 | 0.053 | +34.8 | 38/40 | 0.0000 / 0.0000 |
| 15 | PMEO-E (energy-gated) | -128.6 | 0.942 | 0.1161 | 92.3 | 0.208 | 0.058 | +43.4 | 37/40 | 0.0000 / 0.0000 |
| 16 | MPC-H30 (receding horizon) | -131.7 | 0.939 | 0.1155 | 84.0 | 0.209 | 0.061 | +46.5 | 38/40 | 0.0000 / 0.0000 |
| 17 | Greedy | -143.0 | 0.932 | 0.1188 | 90.1 | 0.732 | 0.068 | +57.7 | 40/40 | 0.0000 / 0.0000 |
| 18 | Full-offload | -160.9 | 0.923 | 0.1294 | 92.8 | 0.256 | 0.077 | +75.7 | 40/40 | 0.0000 / 0.0000 |
| 19 | P-D3QN (2024) | -294.5 | 0.852 | 0.2197 | 90.3 | 0.539 | 0.148 | +209.3 | 40/40 | 0.0000 / 0.0000 |
| 20 | DQN | -296.1 | 0.849 | 0.2125 | 90.0 | 0.534 | 0.151 | +210.9 | 40/40 | 0.0000 / 0.0000 |
| 21 | SAC | -304.1 | 0.840 | 0.1930 | 145.6 | 0.595 | 0.160 | +218.8 | 40/40 | 0.0000 / 0.0000 |
| 22 | TD3 | -336.7 | 0.821 | 0.2085 | 150.9 | 0.495 | 0.179 | +251.5 | 40/40 | 0.0000 / 0.0000 |
| 23 | Random | -339.2 | 0.826 | 0.2337 | 147.2 | 0.499 | 0.174 | +253.9 | 40/40 | 0.0000 / 0.0000 |

## 2. Main results — v2x_hotspot_stress

| rank | method | reward | success | latency (s) | energy | offload | drop | Δ vs PMEO | wins | p (t / Wilcoxon) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PMEO (ours) | -132.4 | 0.957 | 0.0926 | 147.7 | 0.193 | 0.043 |  — | — | — |
| 2 | GDRL (ours) | -132.6 | 0.956 | 0.0926 | 147.2 | 0.193 | 0.044 | +0.2 | 20/40 | 0.5000 / 0.8057 |
| 3 | MPC-H3 (receding horizon) | -134.8 | 0.955 | 0.0928 | 130.3 | 0.194 | 0.045 | +2.4 | 19/40 | 0.0200 / 0.1214 |
| 4 | Predict-TEA | -137.0 | 0.954 | 0.0928 | 147.7 | 0.193 | 0.046 | +4.6 | 37/40 | 0.0000 / 0.0000 |
| 5 | Current-pos exact | -137.1 | 0.954 | 0.0928 | 147.7 | 0.193 | 0.046 | +4.7 | 37/40 | 0.0000 / 0.0000 |
| 6 | Follow-TEA | -140.1 | 0.952 | 0.0922 | 147.9 | 0.194 | 0.048 | +7.7 | 35/40 | 0.0000 / 0.0000 |
| 7 | MPC-H10 (receding horizon) | -147.8 | 0.948 | 0.0937 | 104.6 | 0.193 | 0.052 | +15.4 | 30/40 | 0.0001 / 0.0000 |
| 8 | Deadline-TEA | -162.6 | 0.941 | 0.0951 | 94.6 | 0.193 | 0.059 | +30.2 | 39/40 | 0.0000 / 0.0000 |
| 9 | PPO (BC+KL) | -165.7 | 0.941 | 0.1003 | 148.4 | 0.187 | 0.059 | +33.3 | 40/40 | 0.0000 / 0.0000 |
| 10 | E-Guard TEA | -176.5 | 0.934 | 0.0972 | 92.1 | 0.192 | 0.066 | +44.1 | 39/40 | 0.0000 / 0.0000 |
| 11 | Transformer-PPO (2025) | -178.4 | 0.935 | 0.1024 | 148.0 | 0.207 | 0.065 | +46.0 | 40/40 | 0.0000 / 0.0000 |
| 12 | TEA | -179.9 | 0.932 | 0.0964 | 87.6 | 0.192 | 0.068 | +47.5 | 40/40 | 0.0000 / 0.0000 |
| 13 | Lyapunov | -179.9 | 0.932 | 0.0964 | 87.6 | 0.192 | 0.068 | +47.5 | 40/40 | 0.0000 / 0.0000 |
| 14 | GAT-PPO (2024) | -184.2 | 0.932 | 0.1014 | 148.2 | 0.208 | 0.068 | +51.8 | 40/40 | 0.0000 / 0.0000 |
| 15 | PMEO-E (energy-gated) | -190.1 | 0.927 | 0.0994 | 95.3 | 0.191 | 0.073 | +57.7 | 40/40 | 0.0000 / 0.0000 |
| 16 | MPC-H30 (receding horizon) | -209.2 | 0.918 | 0.1006 | 83.1 | 0.193 | 0.083 | +76.8 | 40/40 | 0.0000 / 0.0000 |
| 17 | Full-offload | -210.2 | 0.918 | 0.1058 | 95.5 | 0.225 | 0.082 | +77.8 | 40/40 | 0.0000 / 0.0000 |
| 18 | Greedy | -213.8 | 0.915 | 0.1014 | 88.9 | 0.746 | 0.085 | +81.4 | 40/40 | 0.0000 / 0.0000 |
| 19 | P-D3QN (2024) | -357.7 | 0.860 | 0.1800 | 91.1 | 0.538 | 0.140 | +225.3 | 40/40 | 0.0000 / 0.0000 |
| 20 | SAC | -414.5 | 0.828 | 0.1666 | 146.2 | 0.572 | 0.172 | +282.1 | 40/40 | 0.0000 / 0.0000 |
| 21 | TD3 | -420.0 | 0.829 | 0.1820 | 151.5 | 0.539 | 0.171 | +287.6 | 40/40 | 0.0000 / 0.0000 |
| 22 | Random | -436.2 | 0.831 | 0.2229 | 149.2 | 0.499 | 0.169 | +303.8 | 40/40 | 0.0000 / 0.0000 |

## 3. Ablation — is trajectory learning necessary?

| difficulty | method | reward | success | latency (s) | energy | Δ (PMEO - variant) | p (paired t) |
|---|---|---|---|---|---|---|---|
| v2x_hotspot_hard | GDRL (ours) | -85.2 | 0.971 | 0.1071 | 147.7 |  — |  — |
| v2x_hotspot_hard | GDRL - traj (expert move) | -85.2 | 0.971 | 0.1071 | 148.2 | +0.0 | 0.3010 |
| v2x_hotspot_hard | GDRL - exact offload | -85.2 | 0.971 | 0.1071 | 147.7 | +0.0 | 0.3235 |

| v2x_hotspot_stress | GDRL (ours) | -132.6 | 0.956 | 0.0926 | 147.2 |  — |  — |
| v2x_hotspot_stress | GDRL - traj (expert move) | -132.3 | 0.957 | 0.0926 | 147.7 | -0.3 | 0.3012 |
| v2x_hotspot_stress | GDRL - exact offload | -132.6 | 0.956 | 0.0926 | 147.2 | +0.0 | nan |

## 3b. Mechanism — decision-order sensitivity (post-move vs current-position offloading)

| difficulty | method | reward | success | latency (s) | Δ vs PMEO |
|---|---|---|---|---|---|
| v2x_hotspot_hard | PMEO (ours) | -85.2 | 0.971 | 0.1072 |  — |
| v2x_hotspot_hard | GDRL (ours) | -85.2 | 0.971 | 0.1071 | +0.0 |
| v2x_hotspot_hard | Current-pos exact | -87.4 | 0.969 | 0.1073 | -2.2 |
| v2x_hotspot_hard | Predict-TEA | -87.4 | 0.969 | 0.1073 | -2.2 |

| v2x_hotspot_stress | PMEO (ours) | -132.4 | 0.957 | 0.0926 |  — |
| v2x_hotspot_stress | GDRL (ours) | -132.6 | 0.956 | 0.0926 | -0.2 |
| v2x_hotspot_stress | Current-pos exact | -137.1 | 0.954 | 0.0928 | -4.7 |
| v2x_hotspot_stress | Predict-TEA | -137.0 | 0.954 | 0.0928 | -4.6 |

## 4. Reproduction

```bash
# proposed method PMEO is training-free: run it directly (per difficulty)
python -m uav_leo_experiment.run_experiment --difficulty v2x_hotspot_hard --episodes 40 --horizon 30 --users 12 --seed 73 --skip_slow --methods postmove_exact,current_exact,predict_tea,follow_tea,mpc_traj_h3 --mpc_horizon 3 --mpc_offload exact --output_dir experiments/uav_leo_v2x_paper_final
# optional: trained residual-PPO variant GDRL (per difficulty)
python -m uav_leo_experiment.train_drl --method gdrl --difficulty v2x_hotspot_hard --steps 40000
python -m uav_leo_experiment.train_drl --method gdrl --difficulty v2x_hotspot_stress --steps 40000
```

Figures: `sweep_reward.png`, `sweep_success_rate.png`, `sweep_latency.png`, `sweep_energy.png`, `sweep_offload_ratio.png`, `sweep_latency_energy_tradeoff.png`, `gdrl_gain_<difficulty>.png` (see `figures/`).
