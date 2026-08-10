@echo off
cd /d C:\Users\Tiffany\Desktop\GDRL\GDRL
"C:\UserFiles\Anaconda3\python.exe" -m uav_leo_experiment.train_drl --method ppo --difficulty v2x_hotspot_stress --users 16 --steps 60000 --lr 3e-4 --eval_every 4000 --rollout_steps 512 --entropy_coef 0.001 --pretrain_steps 12000 --pretrain_expert follow_tea --output_dir experiments/uav_leo_v2x/drl_models --seed 73 > _train_ppo_v2x_hotspot_stress.log 2>&1
