$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host 'Installez les dependances avec .\install.ps1, puis relancez ce fichier.'
    exit 1
}
Write-Host 'Interface : http://127.0.0.1:8765'
Write-Host 'Aucun entrainement ne demarre automatiquement. Utilisez le bouton dans la page.'
& $pythonPath -m slitherai.server --port 8765
