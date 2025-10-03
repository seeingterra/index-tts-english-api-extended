param(
    [string]$VenvPath = ".venv",
    [string]$Requirements = "requirements.txt",
    [string]$LockFile = "requirements-lock.txt",
    [string]$PythonCmd = "",
    [switch]$InstallCpuTorch,
    [switch]$AutoDetectCuda
)

function Find-Python311 {
    $candidates = @(
        "py -3.11",
        "python3.11",
        "python3.11.exe",
        "C:\\Program Files\\Python311\\python.exe",
        "C:\\Python311\\python.exe"
    )
    foreach ($c in $candidates) {
        try {
            $out = & $c -V 2>&1
            if ($out -match "3\.11") { return $c }
        } catch { }
    }
    return $null
}

$pythonExe = ""
$pythonArgs = @()
if ($PythonCmd -ne "") {
    # Support passing a command with args like: 'py -3.11'
    $parts = $PythonCmd -split '\s+'
    $pythonExe = $parts[0]
    if ($parts.Length -gt 1) { $pythonArgs = $parts[1..($parts.Length - 1)] }
    try {
        $out = & $pythonExe @pythonArgs -V 2>&1
        if ($out -match "3\.11") {
            # good
        } else {
            Write-Error "Provided PythonCmd '$PythonCmd' is not Python 3.11 (version: $out)."
            exit 1
        }
    } catch {
        Write-Error "Cannot run provided PythonCmd '$PythonCmd'. Ensure the path is correct and executable."
        exit 1
    }
} else {
    $found = Find-Python311
    if (-not $found) {
        Write-Error "Python 3.11 not found. Install Python 3.11 or make it available as 'py -3.11' / 'python3.11'."
        exit 1
    }
    # If found, the finder returns a command string; split it into exe+args if needed
    $parts = $found -split '\s+'
    $pythonExe = $parts[0]
    if ($parts.Length -gt 1) { $pythonArgs = $parts[1..($parts.Length - 1)] }
}

Write-Host "Using Python command: $pythonExe $([string]::Join(' ', $pythonArgs))"

# Create venv
Write-Host "Creating venv at $VenvPath ..."
$createArgs = @()
if ($pythonArgs) { $createArgs += $pythonArgs }
$createArgs += @('-m','venv',$VenvPath)
& $pythonExe @createArgs

# Activate venv for the remainder of the script
$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Error "Activation script not found at $activate"
    exit 1
}
. $activate

# Upgrade packaging tools
Write-Host "Upgrading pip/setuptools/wheel ..."
python -m pip install --upgrade pip setuptools wheel

# Install all requirements
if (-not (Test-Path $Requirements)) {
    Write-Error "Requirements file '$Requirements' not found."
    exit 1
}

# Optional: check for Microsoft Visual C++ runtime (vcruntime140.dll)
function Test-VCRuntime {
    # Try to find vcruntime140.dll in System32/WinSxS or loaded modules
    $dllNames = @("vcruntime140.dll","msvcp140.dll")
    foreach ($n in $dllNames) {
        $found = Get-ChildItem -Path "$env:SystemRoot\System32\$n" -ErrorAction SilentlyContinue
        if ($found) { return $true }
    }
    # Try checking common Program Files redist install locations in WinSxS (best-effort)
    try {
        $winsxs = Join-Path $env:SystemRoot 'WinSxS'
        if (Test-Path $winsxs) {
            foreach ($n in $dllNames) {
                if (Get-ChildItem -Path $winsxs -Filter $n -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1) {
                    return $true
                }
            }
        }
    } catch { }
    return $false
}

if (-not (Test-VCRuntime)) {
    Write-Warning "Microsoft Visual C++ runtime not detected (vcruntime140.dll/msvcp140.dll). This may cause binary extensions (e.g., PyTorch) to fail to load (WinError 126)."
    Write-Host "If you see WinError 126 when importing torch, install the 'Microsoft Visual C++ Redistributable (2015-2022) x64' from Microsoft:" -ForegroundColor Yellow
    Write-Host "https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist" -ForegroundColor Yellow
}

Write-Host "Installing from $Requirements (this may take a while)..."

# Helper: detect CUDA (nvcc or nvidia-smi) and return wheel tag like +cu121 or +cu120
function Get-CudaWheelTag {
    # Detect CUDA version via nvcc or nvidia-smi, then map to a compatible torch wheel tag.
    # Known compatible torch tags for this project's torch==2.8.* are commonly: +cu121, +cu120, +cu118, +cu117
    $mapping = @{
        '12.1' = '+cu121'
        '12.0' = '+cu120'
        '11.8' = '+cu118'
        '11.7' = '+cu117'
        '11.6' = '+cu116'
    }

    $detect = $null
    try {
        $nvcc = & nvcc --version 2>&1
        if ($nvcc -match "release\s+([0-9]+)\.([0-9]+)") {
            $detect = "$($matches[1]).$($matches[2])"
        }
    } catch { }

    if (-not $detect) {
        try {
            $out = & nvidia-smi 2>&1
            if ($out -match "CUDA Version:\s*([0-9]+)\.([0-9]+)") {
                $detect = "$($matches[1]).$($matches[2])"
            }
        } catch { }
    }

    if (-not $detect) { return $null }

    # Exact map
    if ($mapping.ContainsKey($detect)) { return $mapping[$detect] }

    # Fallback: try to match major.minor to nearest known mapping, prefer same major then closest minor
    $parts = $detect -split '\.'
    if ($parts.Length -lt 2) { return $null }
    $maj = [int]$parts[0]; $min = [int]$parts[1]
    # Find mapping entries with same major
    $candidates = $mapping.Keys | Where-Object { ($_ -split '\.')[0] -eq $maj.ToString() }
    if ($candidates) {
        # choose candidate with minimal minor difference
        $best = $null; $bestDiff = 999
        foreach ($k in $candidates) {
            $km = [int]($k -split '\.')[1]
            $d = [math]::Abs($km - $min)
            if ($d -lt $bestDiff) { $bestDiff = $d; $best = $k }
        }
        if ($best) { return $mapping[$best] }
    }

    return $null
}

# If auto-detect requested, try to detect CUDA and pick GPU wheels; fallback to CPU path if not found
if ($AutoDetectCuda) {
    Write-Host "AutoDetectCuda flag set: attempting to detect installed CUDA version..."
    $cudatag = Get-CudaWheelTag
    if ($cudatag) {
        Write-Host "Detected CUDA wheel tag: $cudatag. Attempting to install matching PyTorch wheels..."
        # Filter requirements to exclude torch/torchaudio
        $tmpReq = Join-Path $env:TEMP "indextts_reqs_no_torch.txt"
        Get-Content $Requirements | Where-Object { 
            ($_ -notmatch '^(\s*#)') -and ($_ -notmatch '^(torch|torchaudio)') 
        } | Out-File -FilePath $tmpReq -Encoding UTF8

        python -m pip install -r $tmpReq
        if ($LASTEXITCODE -ne 0) {
            Write-Error "pip install (filtered) failed. See output above."
            Remove-Item $tmpReq -ErrorAction SilentlyContinue
            exit $LASTEXITCODE
        }

        # Try GPU wheel install; use official torch index for matching tags (best-effort)
        $indexBase = 'https://download.pytorch.org/whl'
        try {
            python -m pip install --index-url "$indexBase/$($cudatag.TrimStart('+'))" "torch==2.8.*$cudatag" "torchaudio==2.8.*$cudatag" -f https://download.pytorch.org/whl/torch_stable.html
        } catch {
            Write-Warning "GPU wheel install failed for tag $cudatag — falling back to CPU wheels."
            python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.*+cpu" "torchaudio==2.8.*+cpu" -f https://download.pytorch.org/whl/torch_stable.html
        }

        Remove-Item $tmpReq -ErrorAction SilentlyContinue
        return
    } else {
        Write-Warning "Unable to detect CUDA via nvcc or nvidia-smi. Falling back to CPU-only torch installation path."
        $InstallCpuTorch = $true
    }
}

# If user requested CPU-only torch wheels, install requirements except torch/torchaudio, then install CPU torch wheels explicitly
if ($InstallCpuTorch) {
    Write-Host "InstallCpuTorch flag set: filtering torch and torchaudio from requirements and installing CPU wheels afterwards..."
    $tmpReq = Join-Path $env:TEMP "indextts_reqs_no_torch.txt"
    Get-Content $Requirements | Where-Object { 
        ($_ -notmatch '^(\s*#)') -and ($_ -notmatch '^(torch|torchaudio)') 
    } | Out-File -FilePath $tmpReq -Encoding UTF8

    python -m pip install -r $tmpReq
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install (filtered) failed. See output above."
        Remove-Item $tmpReq -ErrorAction SilentlyContinue
        exit $LASTEXITCODE
    }

    Write-Host "Installing CPU-only PyTorch and torchaudio wheels (torch==2.8.*+cpu)..."
    python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.*+cpu" "torchaudio==2.8.*+cpu" -f https://download.pytorch.org/whl/torch_stable.html
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install of CPU PyTorch wheels failed. See output above."
        Remove-Item $tmpReq -ErrorAction SilentlyContinue
        exit $LASTEXITCODE
    }

    Remove-Item $tmpReq -ErrorAction SilentlyContinue
} else {
    python -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install failed. See output above."
        exit $LASTEXITCODE
    }
}

# Freeze to lockfile
Write-Host "Generating lockfile $LockFile ..."
python -m pip freeze | Out-File -Encoding UTF8 $LockFile

# Smoke test imports and print versions
Write-Host "Running smoke tests (numpy, torch, transformers, numba, librosa) ..."
$py = @'
import sys, importlib
names = ["numpy","torch","transformers","numba","librosa"]
out = []
for n in names:
    try:
        m = importlib.import_module(n)
        v = getattr(m, "__version__", "unknown")
        out.append(f"{n}={v}")
    except Exception as e:
        print(f"IMPORT-ERROR {n}: {e}", file=sys.stderr)
        sys.exit(2)
print("OK " + " ".join(out))
'@

# Write smoke test to a temporary file and run it to avoid shell quoting issues
$tmp = Join-Path $env:TEMP "indextts_smoke_test.py"
$py | Out-File -FilePath $tmp -Encoding UTF8
& python $tmp
$exit = $LASTEXITCODE
Remove-Item $tmp -ErrorAction SilentlyContinue
if ($exit -ne 0) {
    Write-Error "Smoke test failed."
    exit $exit
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "Smoke test failed."
    exit $LASTEXITCODE
}

Write-Host "Setup complete. Activate the venv with: .\$VenvPath\Scripts\Activate.ps1"
exit 0
