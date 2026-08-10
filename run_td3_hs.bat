@echo off
cd /d C:\Users\Tiffany\Desktop\GDRL\GDRL
C:\UserFiles\Anaconda3\python.exe -m uav_leo_experiment.train_drl --method ppo --difficulty v2x_hotspot_stress --users 16 --steps 60000 --rollout_steps 512 --lr 1e-4 --entropy_coef 0.003 --pretrain_steps 6000 --eval_every 10000 --output_dir experiments/uav_leo_v2x/drl_models --seed 73 > experiments\uav_leo_v2x\drl_models\ppo_train_hs.log 2>&1
