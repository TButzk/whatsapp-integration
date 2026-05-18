param(
    [int]$Port = 8000,
    [string]$BindHost = "0.0.0.0",
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Python da venv não encontrado em .venv\Scripts\python.exe"
}

$EnvFile = Join-Path $ProjectRoot "local_ai_backend\.env"
if (-not (Test-Path $EnvFile)) {
    $EnvExample = Join-Path $ProjectRoot "local_ai_backend\.env.example"
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Write-Host "Arquivo local_ai_backend\.env criado a partir do .env.example"
    }
    else {
        throw "Arquivo local_ai_backend\.env.example não encontrado."
    }
}

$Args = @(
    "-m", "uvicorn",
    "local_ai_backend.main:app",
    "--host", $BindHost,
    "--port", "$Port",
    "--env-file", "local_ai_backend/.env"
)

if ($Reload) {
    $Args += "--reload"
}

& $VenvPython @Args
