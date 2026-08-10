@echo off
cd /d C:\Users\Tiffany\Desktop\GDRL\GDRL
set OMP_NUM_THREADS=6
C:\UserFiles\Anaconda3\python.exe -m uav_leo_experiment.train_drl --method gdrl --difficulty v2x_hotspot_hard --users 12 --steps 40000 --move_std 0.05 --entropy_coef 0.001 --eval_every 4000 --output_dir experiments/uav_leo_v2x/drl_models_v2 --seed 73 > experiments\uav_leo_v2x\train_v2_gdrl_hard2.out.log 2>&1
