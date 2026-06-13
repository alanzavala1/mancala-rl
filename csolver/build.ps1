# Build the C solver into mancala_solver.dll.
#
# Prefers gcc on PATH (the normal case after installing a compiler and
# restarting your shell). Falls back to the WinLibs install location if gcc
# isn't on PATH yet in this session.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) {
    $winlibs = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
        -Filter "BrechtSanders.WinLibs*" -Directory -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($winlibs) {
        $env:Path = "$($winlibs.FullName)\mingw64\bin;$env:Path"
    }
}

$gcc = (Get-Command gcc -ErrorAction SilentlyContinue)
if (-not $gcc) { throw "gcc not found. Install a compiler (e.g. winget install BrechtSanders.WinLibs.POSIX.UCRT) and restart your shell." }

gcc -shared -O2 -Wall -o "$here\mancala_solver.dll" "$here\mancala_solver.c"
if ($?) { Write-Output "built: $here\mancala_solver.dll  (using $($gcc.Source))" }
