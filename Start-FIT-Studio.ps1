param([switch]$Web)
$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$studioPython = Join-Path $projectDir '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $studioPython)) {
    & python -m venv (Join-Path $projectDir '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
}
& $studioPython -c 'import webview, psutil, llmfit, fit_gguf.studio.server' 2>$null
if ($LASTEXITCODE -ne 0) {
    & $studioPython -m pip install -e "${projectDir}[desktop,hardware]"
    if ($LASTEXITCODE -ne 0) { throw 'Studio dependency installation failed.' }
}
$studioMode = if ($Web) { 'gui' } else { 'desktop' }
$studioWorkspace = Join-Path $projectDir 'work/studio'
$studioPythonWindowless = Join-Path $projectDir '.venv/Scripts/pythonw.exe'
$launchOptions = @{ FilePath=$studioPythonWindowless; ArgumentList=@('-m','fit_gguf.cli',$studioMode,'--workspace',('"' + $studioWorkspace + '"')); WorkingDirectory=$projectDir }
if ($Web) { $launchOptions.WindowStyle = 'Hidden' }
Start-Process @launchOptions
