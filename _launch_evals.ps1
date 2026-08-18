param()
$root = "C:\Users\Tiffany\Desktop\GDRL\GDRL"
$script = Join-Path $root "experiments\uav_leo_v2x_paper_final\_eval_mdrl.py"
$env:PYTHONPATH = $root
foreach ($m in @("m_dqn","m_td3","m_sac")) {
    $model = Join-Path $root ("experiments\uav_leo_v2x_paper_final\mdrl_models\k3_" + $m + "_seed1")
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "python"
    $psi.Arguments = '"' + $script + '" --method ' + $m + ' --model_dir "' + $model + '"'
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $true
    $psi.CreateNoWindow = $true
    $psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $proc = [System.Diagnostics.Process]::Start($psi)
    Write-Output ("{0} eval started PID {1}" -f $m, $proc.Id)
}
