$projectRoot = $PSScriptRoot
$pythonExe = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "..\.venv\Scripts\python.exe"))

if (-not (Test-Path $pythonExe)) {
	throw "Python nao encontrado em: $pythonExe"
}

Set-Location $projectRoot
& $pythonExe ".\auto_reply_bridge\app.py"