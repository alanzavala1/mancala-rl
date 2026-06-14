# Build the C policy-inference DLL.
# Run scripts/export_weights.py first to generate cnn/policy_weights.h.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) {
    $w = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
        -Filter "BrechtSanders.WinLibs*" -Directory -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($w) { $env:Path = "$($w.FullName)\mingw64\bin;$env:Path" }
}
if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) { throw "gcc not found." }
if (-not (Test-Path "$here\policy_weights.h")) {
    throw "policy_weights.h missing -- run scripts/export_weights.py first."
}

gcc -shared -O3 -march=native -ffast-math -funroll-loops -o "$here\policy_net.dll" "$here\policy_net.c"
if ($?) { Write-Output "built: $here\policy_net.dll" }
