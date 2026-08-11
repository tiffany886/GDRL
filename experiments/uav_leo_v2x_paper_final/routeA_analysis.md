# Route A analysis — sensitivity and multi-seed robustness

## 1. Sensitivity (v2x_hotspot_hard, 40 episodes, seed 73)
| variant | method | reward | success | latency (s) | energy |
|---|---|---|---|---|---|
| hotspot-hard baseline | PMEO (ours) | -85.2 | 0.971 | 0.1072 | 148.2 |
| hotspot-hard baseline | PMEO-E (energy-gated) | -128.6 | 0.942 | 0.1161 | 92.3 |
| hotspot-hard baseline | Current-pos exact | -87.4 | 0.969 | 0.1073 | 148.2 |
| hotspot-hard baseline | Predict-TEA | -87.4 | 0.969 | 0.1073 | 148.2 |
| hotspot-hard baseline | Follow-TEA | -93.2 | 0.965 | 0.1067 | 148.1 |
| hotspot-hard baseline | MPC-H3 | -85.6 | 0.970 | 0.1074 | 124.0 |
| hotspot-hard baseline | Random | -339.2 | 0.826 | 0.2337 | 147.2 |

| users = 8 | PMEO (ours) | -58.5 | 0.970 | 0.1035 | 147.4 |
| users = 8 | Current-pos exact | -61.2 | 0.967 | 0.1037 | 147.4 |
| users = 8 | Predict-TEA | -61.2 | 0.967 | 0.1037 | 147.4 |
| users = 8 | Follow-TEA | -64.4 | 0.963 | 0.1035 | 147.9 |
| users = 8 | MPC-H3 | -59.8 | 0.967 | 0.1043 | 115.4 |
| users = 8 | Random | -221.7 | 0.830 | 0.2248 | 146.1 |

| users = 24 | PMEO (ours) | -178.1 | 0.969 | 0.1157 | 148.9 |
| users = 24 | Current-pos exact | -183.9 | 0.967 | 0.1159 | 148.9 |
| users = 24 | Predict-TEA | -183.8 | 0.967 | 0.1159 | 148.9 |
| users = 24 | Follow-TEA | -187.4 | 0.965 | 0.1151 | 150.3 |
| users = 24 | MPC-H3 | -181.4 | 0.967 | 0.1160 | 134.3 |
| users = 24 | Random | -733.9 | 0.809 | 0.2482 | 153.5 |

| deadline = 0.50 s | PMEO (ours) | -147.5 | 0.926 | 0.0997 | 147.8 |
| deadline = 0.50 s | Current-pos exact | -153.2 | 0.922 | 0.0999 | 147.8 |
| deadline = 0.50 s | Predict-TEA | -153.1 | 0.922 | 0.0999 | 147.8 |
| deadline = 0.50 s | Follow-TEA | -154.8 | 0.920 | 0.0989 | 147.7 |
| deadline = 0.50 s | MPC-H3 | -147.5 | 0.925 | 0.0994 | 124.9 |
| deadline = 0.50 s | Random | -375.8 | 0.796 | 0.2170 | 147.2 |

| deadline = 0.80 s | PMEO (ours) | -52.7 | 0.994 | 0.1096 | 148.3 |
| deadline = 0.80 s | Current-pos exact | -54.6 | 0.993 | 0.1097 | 148.4 |
| deadline = 0.80 s | Predict-TEA | -54.6 | 0.993 | 0.1097 | 148.4 |
| deadline = 0.80 s | Follow-TEA | -56.6 | 0.991 | 0.1094 | 148.3 |
| deadline = 0.80 s | MPC-H3 | -53.5 | 0.993 | 0.1104 | 116.4 |
| deadline = 0.80 s | Random | -306.9 | 0.852 | 0.2473 | 147.2 |

| hotspot speed = 12 m/s | PMEO (ours) | -102.3 | 0.964 | 0.1283 | 142.6 |
| hotspot speed = 12 m/s | PMEO-E (energy-gated) | -135.2 | 0.942 | 0.1361 | 93.0 |
| hotspot speed = 12 m/s | Current-pos exact | -104.6 | 0.962 | 0.1285 | 142.6 |
| hotspot speed = 12 m/s | Predict-TEA | -104.6 | 0.962 | 0.1285 | 142.6 |
| hotspot speed = 12 m/s | Follow-TEA | -107.2 | 0.961 | 0.1278 | 147.5 |
| hotspot speed = 12 m/s | MPC-H3 | -100.5 | 0.965 | 0.1287 | 120.5 |
| hotspot speed = 12 m/s | Random | -410.0 | 0.789 | 0.2832 | 148.3 |

| hotspot speed = 24 m/s | PMEO (ours) | -78.2 | 0.973 | 0.0967 | 149.0 |
| hotspot speed = 24 m/s | PMEO-E (energy-gated) | -125.6 | 0.941 | 0.1063 | 90.8 |
| hotspot speed = 24 m/s | Current-pos exact | -81.8 | 0.970 | 0.0969 | 149.0 |
| hotspot speed = 24 m/s | Predict-TEA | -81.8 | 0.970 | 0.0969 | 149.0 |
| hotspot speed = 24 m/s | Follow-TEA | -85.0 | 0.968 | 0.0963 | 148.2 |
| hotspot speed = 24 m/s | MPC-H3 | -81.0 | 0.971 | 0.0969 | 124.3 |
| hotspot speed = 24 m/s | Random | -303.1 | 0.844 | 0.2072 | 146.6 |


| energy weight = 0.01 | PMEO (ours) | -125.2 | 0.971 | 0.1072 | 148.2 |
| energy weight = 0.01 | PMEO-E (energy-gated) | -153.8 | 0.941 | 0.1163 | 92.1 |
| energy weight = 0.01 | Current-pos exact | -127.4 | 0.969 | 0.1073 | 148.2 |
| energy weight = 0.01 | Predict-TEA | -127.4 | 0.969 | 0.1073 | 148.2 |
| energy weight = 0.01 | Follow-TEA | -133.1 | 0.965 | 0.1067 | 148.1 |
| energy weight = 0.01 | MPC-H3 | -119.9 | 0.964 | 0.1084 | 98.4 |
| energy weight = 0.01 | Random | -378.9 | 0.826 | 0.2337 | 147.2 |

| energy weight = 0.1 | PMEO (ours) | -525.2 | 0.971 | 0.1079 | 148.0 |
| energy weight = 0.1 | PMEO-E (energy-gated) | -393.3 | 0.926 | 0.1201 | 81.1 |
| energy weight = 0.1 | Current-pos exact | -527.9 | 0.969 | 0.1080 | 148.0 |
| energy weight = 0.1 | Predict-TEA | -527.7 | 0.969 | 0.1077 | 148.0 |
| energy weight = 0.1 | Follow-TEA | -533.0 | 0.965 | 0.1071 | 148.0 |
| energy weight = 0.1 | MPC-H3 | -372.4 | 0.950 | 0.1128 | 86.7 |
| energy weight = 0.1 | Random | -776.3 | 0.826 | 0.2337 | 147.2 |


## 2. Multi-seed robustness (40 episodes per seed × 5 seeds)

### v2x_hotspot_hard

| method | reward (mean of seed means) | 95% CI | p (paired t, pooled over seeds) vs PMEO |
|---|---|---|---|
| PMEO (ours) | -87.46 | ± 14.51 | - |
| PMEO-E (energy-gated) | -128.58 | - | 0.0000 |
| Current-pos exact | -90.59 | ± 14.43 | 0.0000 |
| Predict-TEA | -90.59 | ± 14.43 | 0.0000 |
| Follow-TEA | -93.85 | ± 14.77 | 0.0000 |
| MPC-H3 | -88.81 | ± 13.51 | 0.0013 |
| Random | -356.35 | ± 59.43 | 0.0000 |

### v2x_hotspot_stress

| method | reward (mean of seed means) | 95% CI | p (paired t, pooled over seeds) vs PMEO |
|---|---|---|---|
| PMEO (ours) | -129.63 | ± 7.33 | - |
| PMEO-E (energy-gated) | -190.06 | - | 0.0000 |
| Current-pos exact | -134.65 | ± 7.29 | 0.0000 |
| Predict-TEA | -134.59 | ± 7.31 | 0.0000 |
| Follow-TEA | -136.40 | ± 7.43 | 0.0000 |
| MPC-H3 | -131.82 | ± 6.55 | 0.0001 |
| Random | -453.70 | ± 34.77 | 0.0000 |

