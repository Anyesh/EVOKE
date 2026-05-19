# Elevated payload: adds the Desktop C++ workload (incl. Windows SDK) to VS 2022.
# Run via an elevated scheduled task. The VS installer is a GUI-subsystem app, so
# Start-Process -Wait is required to actually block until the install finishes.
$log = 'C:\Users\User\sdk_modify.log'
"started: $(Get-Date -Format o)" | Out-File $log

$dir = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer'
$exe = $null
foreach ($name in @('vs_installer.exe', 'setup.exe')) {
    $cand = Join-Path $dir $name
    if (Test-Path $cand) { $exe = $cand; break }
}
"installer exe: $exe" | Out-File $log -Append
if (-not $exe) {
    "ERROR: no VS installer exe found" | Out-File $log -Append
    exit 1
}

# single argument string: --installPath is double-quoted so its spaces survive
$argstr = 'modify --installPath "C:\Program Files\Microsoft Visual Studio\2022\Community" --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --quiet --norestart --wait'

try {
    $p = Start-Process -FilePath $exe -ArgumentList $argstr -Wait -PassThru
    "setup exit: $($p.ExitCode)" | Out-File $log -Append
} catch {
    "ERROR: $_" | Out-File $log -Append
}
"finished: $(Get-Date -Format o)" | Out-File $log -Append
