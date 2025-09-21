# Stop uvicorn/webui python processes and remove repo-local .tmp safely
# Usage: Open PowerShell in repo root and run: .\scripts\stop_and_cleanup.ps1

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$cwd = Get-Location

Write-Host "Stopping Python server processes that appear to be this repo's instances..."
# Kill processes whose command line includes this repo path
$pyProcs = Get-Process -Name python -ErrorAction SilentlyContinue
foreach ($p in $pyProcs) {
    try {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
    } catch {
        $cmd = ''
    }
    if ($cmd -and $cmd -like "*$repoRoot*") {
        Write-Host "Stopping PID $($p.Id): $cmd"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
}

# Also kill any process listening on 8010 or 7860 (common dev servers) - only if they belong to python
$ports = @(8010, 7860)
foreach ($port in $ports) {
    $listeners = netstat -aon | Select-String ":$port" | ForEach-Object { ($_ -split '\s+')[-1] } | Sort-Object -Unique
    foreach ($pid in $listeners) {
        try {
            $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($proc -and $proc.ProcessName -eq 'python') {
                Write-Host "Stopping listener PID $pid on port $port"
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            }
        } catch {}
    }
}

# Safe removal of repo-local .tmp directory
$tmpDir = Join-Path $repoRoot '.tmp'
if (Test-Path $tmpDir) {
    Write-Host "Removing temporary directory: $tmpDir"
    Remove-Item -Recurse -Force $tmpDir
} else {
    Write-Host "No temporary directory found at $tmpDir"
}

Write-Host "Cleanup complete."
