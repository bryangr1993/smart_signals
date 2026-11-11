param([ValidateSet('cpu','cuda')][string]$Device = 'cpu')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$signalsRoot = if ($env:SMART_SIGNALS_HOME) { $env:SMART_SIGNALS_HOME } else { Join-Path $env:USERPROFILE '.cache/smart-signals' }
$envPath = Join-Path $signalsRoot 'venv'
$pythonPath = Join-Path $envPath 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & py -3.12 -m venv $envPath
    if ($LASTEXITCODE -ne 0) { throw 'Instale Python 3.12 y vuelva a ejecutar este instalador.' }
}
$torchIndex = if ($Device -eq 'cuda') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
$variant = if ($Device -eq 'cuda') { 'cu128' } else { 'cpu' }
& $pythonPath -m pip install ('torch==2.7.1+' + $variant) ('torchvision==0.22.1+' + $variant) --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'No se pudo instalar PyTorch.' }
& $pythonPath -m pip install -r (Join-Path $repoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias.' }
& $pythonPath (Join-Path $repoRoot 'scripts/prepare_models.py')
if ($LASTEXITCODE -ne 0) { throw 'No se pudo descargar el modelo. Revise la conexión y reintente.' }
Write-Output 'Instalación lista. Ejecute Abrir Smart Signals.cmd.'
