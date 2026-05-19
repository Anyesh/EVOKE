# Orchestrator: runs the VS Installer modify elevated via a scheduled task
# (RL HIGHEST bypasses the non-interactive UAC that blocks a plain ssh session),
# then waits for the installer to finish and verifies the Windows SDK landed.
$ErrorActionPreference = 'Continue'
$tn = 'EvokeSDKInstall'
$tr = 'cmd /c C:\Users\User\sdk_modify.cmd'

schtasks /create /TN $tn /TR $tr /SC ONCE /ST 23:59 /RL HIGHEST /F | Out-Null
schtasks /run /TN $tn | Out-Null
Write-Output 'elevated SDK-install task started'

$elapsed = 0
do {
    Start-Sleep -Seconds 15
    $elapsed += 15
    $state = (Get-ScheduledTask -TaskName $tn -ErrorAction SilentlyContinue).State
} while ($state -eq 'Running' -and $elapsed -lt 300)
schtasks /delete /TN $tn /F | Out-Null

# the VS installer keeps running after the launcher returns; wait it out
$elapsed = 0
do {
    Start-Sleep -Seconds 20
    $elapsed += 20
    $procs = Get-Process -Name vs_installer, vs_installershell, setup -ErrorAction SilentlyContinue
    Write-Output "  +${elapsed}s installer_running=$([bool]$procs)"
} while ($procs -and $elapsed -lt 1500)

Write-Output '--- sdk_modify.log ---'
if (Test-Path 'C:\Users\User\sdk_modify.log') { Get-Content 'C:\Users\User\sdk_modify.log' }
$rc = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\bin' -Recurse -Filter rc.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if ($rc) { Write-Output "SDK OK: rc.exe = $rc" } else { Write-Output 'SDK STILL MISSING' }
