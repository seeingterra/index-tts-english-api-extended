# Start the FastAPI app using the repository .venv and from the repo root
# Usage: .\start_api.ps1

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

# Use UVICORN_PORT env var if present, otherwise default to 8010 to match in-code defaults
$port = $env:UVICORN_PORT; if (-not $port) { $port = 8010 }
Write-Host "Starting FastAPI app (using venv at $venvPath) on port $port..."
 # Force Python to use UTF-8 stdio on Windows so model prints won't raise encoding errors
 $env:PYTHONUTF8 = '1'
 python -m uvicorn fastapi_app.main:app --host 0.0.0.0 --port $port
