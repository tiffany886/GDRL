@echo off
cd /d C:\Users\Tiffany\Desktop\GDRL\GDRL
"C:\UserFiles\Anaconda3\python.exe" -m uav_leo_experiment.train_drl --method td3 --difficulty v2x_hotspot_stress --users 16 --steps 60000 --lr 3e-4 --eval_every 4000  --output_dir experiments/uav_leo_v2x/drl_models --seed 73 > _train_td3_v2x_hotspot_stress.log 2>&1
