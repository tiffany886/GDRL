
param(
    [string[]]$Methods = @("m_td3","m_sac","m_dqn"),
    [int]$Steps = 30000,
    [int]$Seed = 1
)
$root = "C:\Users\Tiffany\Desktop\GDRL\GDRL"
$script = Join-Path $root "experiments\uav_leo_v2x_paper_final\_train_mdrl.py"
$logdir = Join-Path $root "experiments\uav_leo_v2x_paper_final\mdrl_logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$pidFile = Join-Path $logdir "pids.txt"
foreach ($m in $Methods) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "python"
    $psi.Arguments = '"' + $script + '" --preset k3 --method ' + $m + ' --steps ' + $Steps + ' --seed ' + $Seed + ' --output_dir "' + (Join-Path $root "experiments\uav_leo_v2x_paper_final\mdrl_models") + '"'
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $true
    $psi.CreateNoWindow = $true
    $psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $proc = [System.Diagnostics.Process]::Start($psi)
    [System.IO.File]::AppendAllText($pidFile, ($m + " " + $proc.Id + [Environment]::NewLine))
    Write-Output ("{0} started PID {1}" -f $m, $proc.Id)
}
