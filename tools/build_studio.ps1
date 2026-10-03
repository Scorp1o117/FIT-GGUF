param([switch]$OneFile)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$studioPython = Join-Path $projectDir '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $studioPython)) { throw 'Create .venv and install .[desktop,hardware] plus pyinstaller first.' }
$llmfitBinary = Join-Path $projectDir '.venv/Scripts/llmfit.exe'
$buildArgs = @('-m', 'PyInstaller', '--noconfirm', '--clean', '--windowed', '--name', 'FIT-Studio', '--paths', (Join-Path $projectDir 'src'), '--collect-all', 'fit_gguf', '--collect-all', 'webview', '--collect-all', 'pythonnet', '--collect-all', 'clr_loader', '--exclude-module', 'pytest', '--distpath', (Join-Path $projectDir 'dist'), '--workpath', (Join-Path $projectDir 'build/studio'), '--specpath', (Join-Path $projectDir 'build'))
if ($OneFile) { $buildArgs += '--onefile' } else { $buildArgs += '--onedir' }
if (Test-Path -LiteralPath $llmfitBinary) { $buildArgs += @('--add-binary', "$llmfitBinary;.") }
$buildArgs += (Join-Path $projectDir 'tools/studio_entry.py')
& $studioPython @buildArgs
if ($LASTEXITCODE -ne 0) { throw 'Desktop build failed.' }
$appDirectory = if ($OneFile) { Join-Path $projectDir 'dist' } else { Join-Path $projectDir 'dist/FIT-Studio' }
Copy-Item -LiteralPath (Join-Path $projectDir 'LICENSE') -Destination $appDirectory
Copy-Item -LiteralPath (Join-Path $projectDir 'THIRD_PARTY_NOTICES.md') -Destination $appDirectory
Copy-Item -LiteralPath (Join-Path $projectDir 'docs/studio-windows.md') -Destination (Join-Path $appDirectory 'START-HERE.md')
& $studioPython (Join-Path $projectDir 'tools/collect_studio_licenses.py') (Join-Path $appDirectory 'licenses')
if ($LASTEXITCODE -ne 0) { throw 'License collection failed.' }
Write-Host 'Desktop build created in dist/FIT-Studio. Requires Windows Edge WebView2 runtime.'
