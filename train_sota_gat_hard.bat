@echo off
cd /d C:\Users\Tiffany\Desktop\GDRL\GDRL
set OMP_NUM_THREADS=3
set MKL_NUM_THREADS=3
"C:\UserFiles\Anaconda3\python.exe" -m uav_leo_experiment.train_drl --method gat_ppo --difficulty v2x_hotspot_hard --users 12 --steps 40000 --lr 3e-4 --eval_every 4000 --rollout_steps 256 --output_dir experiments/uav_leo_v2x/drl_models_v2 --seed 73 --pretrain_steps 12000 --pretrain_expert follow_tea --kl_coef 0.05 --critic_warmup_updates 10 --entropy_coef 0.003 > train_sota_gat_hard.log 2>&1
