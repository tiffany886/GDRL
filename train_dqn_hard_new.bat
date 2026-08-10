@echo off
cd /d C:\Users\Tiffany\Desktop\GDRL\GDRL
C:\UserFiles\Anaconda3\python.exe -m uav_leo_experiment.train_drl --method dqn --difficulty v2x_hotspot_hard --users 12 --steps 40000 --lr 1e-3 --eval_every 4000 --output_dir experiments/uav_leo_v2x/drl_models --seed 73 > _train_dqn_v2x_hotspot_hard_new.log 2>&1
