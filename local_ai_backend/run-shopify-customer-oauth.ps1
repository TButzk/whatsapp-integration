param(
    [string]$ApiBaseUrl = "http://localhost:8000",
    [string]$EnvPath = "",
    [string]$RedirectUri = "",
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($EnvPath)) {
    $EnvPath = Join-Path $PSScriptRoot ".env"
}

function Read-EnvFile {
    param([string]$Path)

    $map = [ordered]@{}
    if (-not (Test-Path -Path $Path)) {
        throw "Arquivo .env nao encontrado em: $Path"
    }

    $lines = Get-Content -Path $Path
    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line.TrimStart().StartsWith("#")) { continue }

        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { continue }

        $key = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1)
        $map[$key] = $value
    }
    return $map
}

function Get-EnvValue {
    param(
        [System.Collections.IDictionary]$Env,
        [string]$Key
    )

    if ($Env.Contains($Key)) {
        $value = $Env[$Key]
        if ($null -eq $value) {
            return ""
        }
        return [string]$value
    }
    return ""
}

function Upsert-EnvValue {
    param(
        [string]$Path,
        [string]$Key,
        [string]$Value
    )

    $lines = @()
    if (Test-Path -Path $Path) {
        $lines = Get-Content -Path $Path
    }

    $escapedKey = [regex]::Escape($Key)
    $pattern = "^$escapedKey="
    $newLine = "$Key=$Value"
    $updated = $false

    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $pattern) {
            $lines[$i] = $newLine
            $updated = $true
            break
        }
    }

    if (-not $updated) {
        if ($lines.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($lines[-1])) {
            $lines += ""
        }
        $lines += $newLine
    }

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($Path, $lines, $encoding)
}

function Get-ConfiguredAuthEndpoint {
    param([System.Collections.IDictionary]$Env)

    $auth = ""
    $auth = (Get-EnvValue -Env $Env -Key "SHOPIFY_CUSTOMER_OAUTH_AUTHORIZATION_ENDPOINT").Trim()
    if (-not [string]::IsNullOrWhiteSpace($auth)) {
        return $auth
    }

    $token = ""
    $token = (Get-EnvValue -Env $Env -Key "SHOPIFY_CUSTOMER_OAUTH_TOKEN_ENDPOINT").Trim()
    if (-not [string]::IsNullOrWhiteSpace($token)) {
        return $token -replace "/oauth/token$", "/oauth/authorize"
    }

    return ""
}

Write-Host "[1/5] Lendo configuracao de $EnvPath"
$envMap = Read-EnvFile -Path $EnvPath

$clientId = (Get-EnvValue -Env $envMap -Key "SHOPIFY_CUSTOMER_CLIENT_ID").Trim()
$redirectUri = (Get-EnvValue -Env $envMap -Key "SHOPIFY_CUSTOMER_REDIRECT_URI").Trim()
$lookupEnabled = (Get-EnvValue -Env $envMap -Key "SHOPIFY_CUSTOMER_LOOKUP_ENABLED").Trim()
$authEndpoint = Get-ConfiguredAuthEndpoint -Env $envMap

if (-not [string]::IsNullOrWhiteSpace($RedirectUri)) {
    $redirectUri = $RedirectUri.Trim()
    Upsert-EnvValue -Path $EnvPath -Key "SHOPIFY_CUSTOMER_REDIRECT_URI" -Value $redirectUri
}

if ([string]::IsNullOrWhiteSpace($lookupEnabled)) {
    Upsert-EnvValue -Path $EnvPath -Key "SHOPIFY_CUSTOMER_LOOKUP_ENABLED" -Value "true"
    $lookupEnabled = "true"
}

if ([string]::IsNullOrWhiteSpace($clientId)) {
    throw "SHOPIFY_CUSTOMER_CLIENT_ID esta vazio no .env. Preencha este valor e execute novamente."
}

if ([string]::IsNullOrWhiteSpace($redirectUri)) {
    $redirectUri = Read-Host "SHOPIFY_CUSTOMER_REDIRECT_URI nao definido. Informe a URL de callback (ex: https://app.tiarlesbutzk.com.br/admin/shopify/customer-oauth/callback)"
    if ([string]::IsNullOrWhiteSpace($redirectUri)) {
        throw "SHOPIFY_CUSTOMER_REDIRECT_URI e obrigatorio."
    }
    Upsert-EnvValue -Path $EnvPath -Key "SHOPIFY_CUSTOMER_REDIRECT_URI" -Value $redirectUri
}

if ([string]::IsNullOrWhiteSpace($authEndpoint)) {
    throw "Nao foi possivel inferir endpoint de autorizacao. Defina SHOPIFY_CUSTOMER_OAUTH_TOKEN_ENDPOINT ou SHOPIFY_CUSTOMER_OAUTH_AUTHORIZATION_ENDPOINT no .env."
}

Write-Host "[2/5] Solicitando authorize-url no backend"
$encodedRedirect = [System.Uri]::EscapeDataString($redirectUri)
$authorizeUrl = "$($ApiBaseUrl.TrimEnd('/'))/admin/shopify/customer-oauth/authorize-url?redirect_uri=$encodedRedirect"
try {
    $authorizeResponse = Invoke-RestMethod -Method GET -Uri $authorizeUrl
} catch {
    $body = $_.ErrorDetails.Message
    if ($body -and $body -match "shopify_customer_oauth_not_configured") {
        throw "Backend retornou shopify_customer_oauth_not_configured. Verifique no .env: SHOPIFY_CUSTOMER_CLIENT_ID, SHOPIFY_CUSTOMER_REDIRECT_URI e endpoint OAuth."
    }
    throw "Falha ao chamar $authorizeUrl. Erro: $($_.Exception.Message) Body: $body"
}

Add-Type -AssemblyName System.Web
$authUri = [System.Uri]$authorizeResponse.authorization_url
$authQuery = [System.Web.HttpUtility]::ParseQueryString($authUri.Query)
$runtimeClientId = ""
if ($null -ne $authQuery["client_id"]) {
    $runtimeClientId = [string]$authQuery["client_id"]
}
$runtimeClientId = $runtimeClientId.Trim()

$runtimeRedirectUri = ""
if ($null -ne $authQuery["redirect_uri"]) {
    $runtimeRedirectUri = [string]$authQuery["redirect_uri"]
}
$runtimeRedirectUri = $runtimeRedirectUri.Trim()

if (-not [string]::Equals($runtimeClientId, $clientId, [System.StringComparison]::Ordinal)) {
    throw "Backend em execucao usa CLIENT_ID antigo ($runtimeClientId), mas o .env atual tem $clientId. Reinicie a API para recarregar o .env e execute novamente."
}

if (-not [string]::Equals($runtimeRedirectUri, $redirectUri, [System.StringComparison]::Ordinal)) {
    throw "Backend em execucao usa redirect_uri diferente ($runtimeRedirectUri), mas o script/.env usa $redirectUri. Reinicie a API e tente novamente."
}

Write-Host "[3/5] URL de autorizacao recebida"
Write-Host "state: $($authorizeResponse.state)"
Write-Host "redirect_uri: $($authorizeResponse.redirect_uri)"

if (-not $NoBrowser) {
    Write-Host "Abrindo navegador para autorizacao Shopify..."
    Start-Process $authorizeResponse.authorization_url | Out-Null
}

$code = Read-Host "Cole o parametro 'code' retornado no callback"
if ([string]::IsNullOrWhiteSpace($code)) {
    throw "Authorization code vazio."
}

Write-Host "[4/5] Trocando code por token"
$exchangeUrl = "$($ApiBaseUrl.TrimEnd('/'))/admin/shopify/customer-oauth/exchange-code"
$exchangeBody = @{
    code = $code
    redirect_uri = $authorizeResponse.redirect_uri
    code_verifier = $authorizeResponse.code_verifier
} | ConvertTo-Json

try {
    $exchangeResponse = Invoke-RestMethod -Method POST -Uri $exchangeUrl -ContentType "application/json" -Body $exchangeBody
} catch {
    $body = $_.ErrorDetails.Message
    throw "Falha ao trocar authorization code. Erro: $($_.Exception.Message) Body: $body"
}

if (-not $exchangeResponse.ok) {
    throw "Exchange OAuth retornou ok=false"
}

if ($exchangeResponse.access_token) {
    Upsert-EnvValue -Path $EnvPath -Key "SHOPIFY_CUSTOMER_MCP_TOKEN" -Value $exchangeResponse.access_token
}
if ($exchangeResponse.refresh_token) {
    Upsert-EnvValue -Path $EnvPath -Key "SHOPIFY_CUSTOMER_REFRESH_TOKEN" -Value $exchangeResponse.refresh_token
}

Write-Host "[5/5] Concluido"
Write-Host "Tokens salvos no .env:"
Write-Host "- SHOPIFY_CUSTOMER_MCP_TOKEN: $([string]::IsNullOrWhiteSpace($exchangeResponse.access_token) -eq $false)"
Write-Host "- SHOPIFY_CUSTOMER_REFRESH_TOKEN: $([string]::IsNullOrWhiteSpace($exchangeResponse.refresh_token) -eq $false)"
Write-Host "Reinicie a API para carregar variaveis atualizadas."
