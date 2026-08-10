@echo off
cd /d C:\Users\Tiffany\Desktop\GDRL\GDRL
set OMP_NUM_THREADS=3
set MKL_NUM_THREADS=3
"C:\UserFiles\Anaconda3\python.exe" -m uav_leo_experiment.train_drl --method d3qn --difficulty v2x_hotspot_stress --users 16 --steps 60000 --lr 3e-4 --eval_every 4000 --rollout_steps 256 --output_dir experiments/uav_leo_v2x/drl_models_v2 --seed 73  > train_sota_d3qn_stress.log 2>&1
