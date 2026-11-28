# LEO installer — Windows (PowerShell 5.1+).
#
#   irm <raw install.ps1 URL> | iex
#
# This script never downloads or runs a third-party installer. It uses an
# existing Python installation and git only.
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE '.local\share\leo-agent'),
    [string]$RepoUrl = 'https://github.com/shoaibyazdani/leo-agent.git',
    [switch]$SkipOnboard
)

$ErrorActionPreference = 'Stop'

if (-not $IsWindows -and $env:OS -ne 'Windows_NT') {
    throw 'install.ps1 is for Windows; use install.sh on macOS or Linux.'
}

function Test-LeoCheckout([string]$Directory) {
    $project = Join-Path $Directory 'pyproject.toml'
    return (Test-Path $project) -and (Select-String -Path $project -Pattern '^name = "leo-agent-os"' -Quiet)
}

if (Test-LeoCheckout $PSScriptRoot) {
    $repoDir = $PSScriptRoot
} elseif (Test-LeoCheckout $InstallDir) {
    $repoDir = $InstallDir
} else {
    Write-Host '-> fetching LEO source'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $InstallDir) | Out-Null
    git clone --depth 1 $RepoUrl $InstallDir
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed.' }
    $repoDir = $InstallDir
}
Push-Location $repoDir

$pythonExe = $null
$pythonPrefix = @()
foreach ($candidate in @(@('py', '-3.12'), @('py', '-3.11'), @('python', ''))) {
    $exe = Get-Command $candidate[0] -ErrorAction SilentlyContinue
    if (-not $exe) { continue }
    $prefix = if ($candidate[1]) { @($candidate[1]) } else { @() }
    & $exe.Source @prefix -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonExe = $exe.Source
        $pythonPrefix = $prefix
        break
    }
}
if (-not $pythonExe) { throw 'Python 3.11+ is required; install it from python.org and retry.' }

$venvPython = Join-Path $repoDir '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host '-> creating .venv'
    & $pythonExe @pythonPrefix -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }
}
Write-Host '-> installing LEO'
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw 'LEO installation failed.' }

New-Item -ItemType Directory -Force -Path (Join-Path $env:USERPROFILE '.config\leo') | Out-Null
Write-Host '-> initializing local state'
& $venvPython -m leo.interfaces.cli init | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'LEO initialization failed.' }

$binDir = if ($env:LEO_BIN_DIR) { $env:LEO_BIN_DIR } else { Join-Path $env:USERPROFILE '.local\bin' }
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

# The Python shim loads the private env file before invoking the CLI, so keys
# never appear in a command line, a config file or the launcher text.
$shim = Join-Path $binDir 'leo.py'
$shimText = @'
import os
import sys

repo = "__REPO__"
sys.path.insert(0, repo)
os.chdir(repo)
from leo.onboarding_wizard import apply_env_file

apply_env_file()
from leo.interfaces.cli import main

main()
'@.Replace('__REPO__', $repoDir.Replace('\', '\\'))
Set-Content -Path $shim -Value $shimText -Encoding UTF8

$launcher = Join-Path $binDir 'leo.cmd'
Set-Content -Path $launcher -Encoding ASCII -Value @(
    '@echo off',
    "set `"LEO_PY=$venvPython`"",
    "set `"LEO_SHIM=$shim`"",
    '"%LEO_PY%" "%LEO_SHIM%" %*'
)

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -notlike "*$binDir*") {
    [Environment]::SetEnvironmentVariable('Path', "$userPath;$binDir", 'User')
    Write-Host "-> added $binDir to your user PATH"
}
Write-Host "-> installed $launcher"

Pop-Location
if ($SkipOnboard) {
    Write-Host '-> onboarding skipped; run: leo onboard'
} else {
    Write-Host '-> starting onboarding'
    & $venvPython -m leo.interfaces.cli onboard
}
