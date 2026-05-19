# Incremental rebuild of the EVOKE llama.cpp fork on gpu-host (ninja picks up
# only the changed sources). Assumes build/ is already configured.
$ErrorActionPreference = 'Continue'

$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

$tooldir = (uv tool dir).Trim()
$cmakeExe = (Get-ChildItem $tooldir -Recurse -Filter cmake.exe -ErrorAction SilentlyContinue | Select-Object -First 1).FullName

Set-Location 'C:\Users\User\llama.cpp'
& $cmakeExe --build build --target llama -j
if ($LASTEXITCODE -ne 0) { Write-Output 'REBUILD FAILED'; exit 1 }
Write-Output 'REBUILD OK'
