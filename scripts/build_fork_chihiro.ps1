# Build the EVOKE llama.cpp fork on gpu-host with CUDA.
# Imports the MSVC environment, pulls cmake + ninja via uv, configures and builds libllama.
$ErrorActionPreference = 'Continue'

$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { Write-Output 'ERROR: vcvars64.bat not found'; exit 1 }

Write-Output '[1/4] importing MSVC environment'
cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}
$cl = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
Write-Output "  cl.exe: $cl"
if (-not $cl) { Write-Output 'ERROR: cl.exe not on PATH after vcvars'; exit 1 }

Write-Output '[2/4] ensuring cmake + ninja via uv'
uv tool install cmake 2>&1 | Out-Null
uv tool install ninja 2>&1 | Out-Null
$tooldir = (uv tool dir).Trim()
$cmakeExe = (Get-ChildItem $tooldir -Recurse -Filter cmake.exe -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
$ninjaExe = (Get-ChildItem $tooldir -Recurse -Filter ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
Write-Output "  cmake: $cmakeExe"
Write-Output "  ninja: $ninjaExe"
if (-not $cmakeExe -or -not $ninjaExe) { Write-Output 'ERROR: cmake/ninja not found under uv tools'; exit 1 }

Set-Location 'C:\Users\User\llama.cpp'
if (Test-Path build) { Remove-Item build -Recurse -Force }

Write-Output '[3/4] configuring (CUDA, shared libs)'
& $cmakeExe -B build -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninjaExe" -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF `
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
if ($LASTEXITCODE -ne 0) { Write-Output 'ERROR: configure failed'; exit 1 }

Write-Output '[4/4] building llama'
& $cmakeExe --build build --target llama -j
if ($LASTEXITCODE -ne 0) { Write-Output 'ERROR: build failed'; exit 1 }

Write-Output 'BUILD OK'
Get-ChildItem build -Recurse -Filter '*.dll' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
