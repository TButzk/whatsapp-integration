param(
    [int]$TimeoutSeconds = 180,
    [string]$LocalUrl = "http://localhost:65271/",
    [string]$PublicUrl = "https://app.tiarlesbutzk.com.br/",
    [switch]$SkipPublicCheck
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$Message) {
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Wait-HttpOk(
    [string]$Url,
    [int]$TimeoutSeconds
) {
    $start = Get-Date
    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        try {
            $status = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 10 -Uri $Url).StatusCode
            if ($status -eq 200) {
                Write-Info "Endpoint OK: $Url (200)"
                return
            }
        }
        catch {
            # Retry until timeout.
        }
        Start-Sleep -Seconds 2
    }

    throw "Timeout aguardando endpoint responder 200: $Url"
}

function Wait-ServiceHealthy(
    [string]$ServiceName,
    [int]$TimeoutSeconds
) {
    $start = Get-Date
    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        $id = (docker compose ps -q $ServiceName 2>$null)
        if ($id) {
            $id = $id.Trim()
            $state = (docker inspect --format "{{.State.Status}}" $id 2>$null)
            $health = (docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}" $id 2>$null)
            if ($state -eq "running" -and ($health -eq "healthy" -or $health -eq "no-healthcheck")) {
                Write-Info "$ServiceName pronto (state=$state, health=$health)."
                return
            }
        }
        Start-Sleep -Seconds 2
    }

    throw "Timeout aguardando servico saudavel: $ServiceName"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

Write-Info "Iniciando self-heal da stack WhatsApp..."

docker compose up -d whatsapp-postgres whatsapp-redis | Out-Host
Wait-ServiceHealthy -ServiceName "whatsapp-postgres" -TimeoutSeconds $TimeoutSeconds
Wait-ServiceHealthy -ServiceName "whatsapp-redis" -TimeoutSeconds $TimeoutSeconds

docker compose up -d whatsapp-chatwoot-web whatsapp-nginx-gateway whatsapp-cloudflared | Out-Host
Wait-ServiceHealthy -ServiceName "whatsapp-chatwoot-web" -TimeoutSeconds $TimeoutSeconds

Write-Info "Validando endpoint local..."
Wait-HttpOk -Url $LocalUrl -TimeoutSeconds $TimeoutSeconds

if (-not $SkipPublicCheck) {
    Write-Info "Validando endpoint publico..."
    Wait-HttpOk -Url $PublicUrl -TimeoutSeconds $TimeoutSeconds
}
else {
    Write-Warn "Check publico ignorado por -SkipPublicCheck."
}

Write-Host ""
Write-Info "Self-heal concluido com sucesso."