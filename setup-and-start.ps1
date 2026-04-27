param(
    [switch]$SkipPull,
    [switch]$SkipPrune,
    [switch]$StartBridge
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$Message) {
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Err([string]$Message) {
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Assert-Command([string]$CommandName, [string]$InstallHint) {
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Comando '$CommandName' nao encontrado. $InstallHint"
    }
}

function Wait-ContainerReady(
    [string]$ContainerName,
    [int]$TimeoutSeconds = 240
) {
    $start = Get-Date

    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        $state = (docker inspect --format "{{.State.Status}}" $ContainerName 2>$null)
        if (-not $state) {
            Start-Sleep -Seconds 2
            continue
        }

        $healthRaw = (docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}" $ContainerName 2>$null)

        if ($state -ne "running") {
            Start-Sleep -Seconds 2
            continue
        }

        if ($healthRaw -eq "healthy" -or $healthRaw -eq "no-healthcheck") {
            Write-Info "$ContainerName pronto (state=$state, health=$healthRaw)."
            return
        }

        if ($healthRaw -eq "unhealthy") {
            throw "$ContainerName ficou unhealthy. Verifique logs com: docker compose logs --tail 120"
        }

        Start-Sleep -Seconds 3
    }

    throw "Timeout aguardando $ContainerName ficar pronto."
}

function Get-ChatwootWithRetry([int]$Attempts = 3) {
    for ($i = 1; $i -le $Attempts; $i++) {
        Write-Info "Tentativa $i/$Attempts de pull da imagem chatwoot/chatwoot:latest-ce"
        try {
            docker pull --platform linux/amd64 chatwoot/chatwoot:latest-ce | Out-Host
            Write-Info "Pull do Chatwoot concluido."
            return
        }
        catch {
            if ($i -eq $Attempts) {
                throw "Nao foi possivel baixar a imagem do Chatwoot apos $Attempts tentativas."
            }

            Write-Warn "Falha no pull da imagem do Chatwoot. Tentando limpeza de cache antes de nova tentativa..."

            if (-not $SkipPrune) {
                docker builder prune -af | Out-Host
                docker image prune -f | Out-Host
            }
        }
    }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

Write-Info "Iniciando setup e start dos servidores..."

Assert-Command -CommandName "docker" -InstallHint "Instale o Docker Desktop para Windows."

Write-Info "Validando Docker daemon..."
docker info | Out-Null

docker compose version | Out-Null

if (-not (Test-Path ".env")) {
    throw "Arquivo .env nao encontrado em $scriptRoot. Copie .env.example para .env e preencha as variaveis."
}

if (-not $SkipPull) {
    Get-ChatwootWithRetry

    Write-Info "Baixando imagens restantes do compose..."
    docker compose pull postgres redis nginx cloudflared | Out-Host
}

Write-Info "Subindo dependencias de banco/cache..."
docker compose up -d postgres redis apptrip-postgres | Out-Host

Write-Info "Executando preparacao do banco do Chatwoot (db:chatwoot_prepare)..."
docker compose run --rm -e RAILS_ENV=production chatwoot bundle exec rails db:chatwoot_prepare | Out-Host

Write-Info "Subindo todos os servicos com build quando necessario..."
docker compose up -d --build | Out-Host

Write-Info "Reiniciando Chatwoot e Sidekiq para garantir schema atualizado..."
docker compose restart chatwoot sidekiq | Out-Host

Write-Info "Aguardando servicos principais ficarem prontos..."
Wait-ContainerReady -ContainerName "whatsapp-integration-postgres-1" -TimeoutSeconds 180
Wait-ContainerReady -ContainerName "whatsapp-integration-redis-1" -TimeoutSeconds 180
Wait-ContainerReady -ContainerName "whatsapp-integration-chatwoot-1" -TimeoutSeconds 300

if ($StartBridge) {
    $bridgeDir = Join-Path $scriptRoot "auto_reply_bridge"
    $bridgeVenvActivate = Join-Path $bridgeDir ".venv\Scripts\Activate.ps1"
    $bridgeApp = Join-Path $bridgeDir "app.py"

    if ((Test-Path $bridgeDir) -and (Test-Path $bridgeVenvActivate) -and (Test-Path $bridgeApp)) {
        Write-Info "Iniciando auto_reply_bridge em nova janela PowerShell..."
        $bridgeCmd = "Set-Location '$bridgeDir'; . '$bridgeVenvActivate'; python app.py"
        Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $bridgeCmd | Out-Null
    }
    else {
        Write-Warn "Bridge nao iniciado: faltou auto_reply_bridge/.venv ou app.py."
    }
}

Write-Info "Status final dos containers:"
docker compose ps -a | Out-Host

Write-Host ""
Write-Info "Setup e start concluidos."
Write-Info "App exposto em: http://localhost:65271"
Write-Info "Logs: docker compose logs -f"
Write-Info "Parar tudo: docker compose down"
