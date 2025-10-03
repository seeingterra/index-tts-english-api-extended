# Start the Web UI script inside the repository .venv and from the repo root
# Usage: .\start_webui.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location (Join-Path $scriptDir "..")

$venvPath = Join-Path $PWD ".venv"
if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath"
    python -m venv $venvPath
}

# Activate the repo venv
$activate = Join-Path $venvPath "Scripts\Activate.ps1"
. $activate

# Ensure pip is up-to-date and install requirements from the repo root if they exist
python -m pip install --upgrade pip
if (Test-Path (Join-Path $PWD "requirements.txt")) {
    python -m pip install -r (Join-Path $PWD "requirements.txt")
}

Write-Host "Starting Web UI (using venv at $venvPath)..."
# Force Python to use UTF-8 stdio on Windows so model prints won't raise encoding errors
$env:PYTHONUTF8 = '1'
python (Join-Path $PWD "webui.py") --port 7860 --host 127.0.0.1
