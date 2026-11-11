param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$signalsRoot = if ($env:SMART_SIGNALS_HOME) { $env:SMART_SIGNALS_HOME } else { Join-Path $env:USERPROFILE '.cache/smart-signals' }
$pythonPath = Join-Path $signalsRoot 'venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Ejecute primero scripts/install.ps1.' }
$ready = $false
try {
    $state = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/status' -TimeoutSec 2
    $ready = $null -ne $state.state
} catch { }
if (-not $ready) {
    $logDir = Join-Path $signalsRoot 'logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    Start-Process -FilePath $pythonPath -ArgumentList @('app.py') -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'server.out.log') -RedirectStandardError (Join-Path $logDir 'server.err.log')
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $state = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/status' -TimeoutSec 1
            if ($null -ne $state.state) { $ready = $true; break }
        } catch { }
    }
}
if (-not $ready) { throw ('No se inició la aplicación. Revise ' + $signalsRoot + '/logs/server.err.log') }
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8765' }
