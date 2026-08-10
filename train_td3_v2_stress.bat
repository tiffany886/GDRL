@echo off
cd /d C:\Users\Tiffany\Desktop\GDRL\GDRL
set OMP_NUM_THREADS=4
C:\UserFiles\Anaconda3\python.exe -m uav_leo_experiment.train_drl --method td3 --difficulty v2x_hotspot_stress --users 16 --steps 30000 --lr 3e-4 --eval_every 3000 --output_dir experiments/uav_leo_v2x/drl_models_v2 --seed 73 > experiments\uav_leo_v2x\train_v2_td3_stress.out.log 2>&1
