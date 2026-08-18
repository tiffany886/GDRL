param()
$root = "C:\Users\Tiffany\Desktop\GDRL\GDRL"
$script = Join-Path $root "experiments\uav_leo_v2x_paper_final\_run_ablation_energy.py"
$logdir = Join-Path $root "experiments\uav_leo_v2x_paper_final\mdrl_logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$env:PYTHONPATH = $root
foreach ($mode in @("ablation","energy")) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "python"
    $psi.Arguments = '"' + $script + '" --mode ' + $mode
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $true
    $psi.CreateNoWindow = $true
    $psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $proc = [System.Diagnostics.Process]::Start($psi)
    Write-Output ("{0} started PID {1}" -f $mode, $proc.Id)
}
