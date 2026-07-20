# Runnable setup on this machine

This project was verified with the existing `pytorch-cpu` conda environment.
The original `GDRL` environment still fails while importing PyTorch because of
a Windows DLL loading error.

## Verified environment

- Python: 3.9.16
- torch: 2.1.0
- torch-geometric: 2.4.0
- gymnasium: 0.29.1
- stable-baselines3: 2.2.1
- sb3-contrib: 2.2.1
- numpy: 1.23.5
- scipy: 1.11.3
- tensorboard: 2.21.0

## Run

```powershell
conda activate pytorch-cpu
cd C:\Users\Tiffany\Desktop\GDRL\GDRL
python main.py
```

For a quick smoke test:

```powershell
python main.py --total_step 1 --T 2
```

## Channel model

The default channel mode is `paper_approx`. It is a Python approximation of
the paper's Eq. (1)-(3): path loss, environmental log-normal attenuation,
Rician small-scale fading, and UPA array response.

Available modes:

```powershell
$env:GDRL_CHANNEL_MODE = "paper_approx"
$env:GDRL_CHANNEL_MODE = "simple"
$env:GDRL_CHANNEL_MODE = "matlab_p681"
```

The installed MATLAB is R2017a. Its Python engine only supports Python 2.7,
3.4, and 3.5, so it cannot be imported from the verified Python 3.9
environment. `matlab_p681` requires a newer MATLAB release whose Python engine
supports the training environment.

For strict reproduction of the paper's channel model, install a newer MATLAB
release whose Python engine supports the Python version used for training.
