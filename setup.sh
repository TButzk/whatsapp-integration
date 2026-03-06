#!/usr/bin/env bash
# =============================================================================
# setup.sh — Inicialização do Chatwoot Community Edition
#
# Uso:
#   chmod +x setup.sh
#   ./setup.sh
#
# O que este script faz:
#   1. Verifica pré-requisitos (Docker, Docker Compose, arquivo .env)
#   2. (Opcional) Configura Swap de 1 GB no host se ainda não existir
#   3. Faz o pull das imagens Docker
#   4. Inicializa / migra o banco de dados
#   5. Sobe todos os serviços em modo daemon
#
# ATENÇÃO: Execute como usuário com permissão para rodar Docker.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Cores para output
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # Sem cor

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# 1. Verificar pré-requisitos
# ---------------------------------------------------------------------------
info "Verificando pré-requisitos..."

if ! command -v docker &>/dev/null; then
    error "Docker não encontrado. Instale com: curl -fsSL https://get.docker.com | sh"
    exit 1
fi

if ! docker compose version &>/dev/null; then
    error "Docker Compose v2 não encontrado. Atualize o Docker ou instale o plugin."
    exit 1
fi

if [ ! -f ".env" ]; then
    error "Arquivo .env não encontrado."
    error "Execute:  cp .env.example .env   e preencha os valores."
    exit 1
fi

info "Pré-requisitos OK."

# ---------------------------------------------------------------------------
# 2. Configurar Swap (recomendado para o Sidekiq)
# ATENÇÃO: O Sidekiq apresenta vazamentos graduais de memória sob uso intenso.
# Um Swap de pelo menos 1 GB no SO host evita kills por OOM (Out Of Memory).
# ---------------------------------------------------------------------------
SWAPFILE="/swapfile"
if [ ! -f "$SWAPFILE" ]; then
    warn "Swap não detectado em $SWAPFILE."
    read -r -p "Deseja criar 1 GB de Swap agora? [s/N] " CREATE_SWAP
    if [[ "${CREATE_SWAP:-N}" =~ ^[Ss]$ ]]; then
        info "Criando Swap de 1 GB em $SWAPFILE..."
        sudo fallocate -l 1G "$SWAPFILE"
        sudo chmod 600 "$SWAPFILE"
        sudo mkswap "$SWAPFILE"
        sudo swapon "$SWAPFILE"
        # Persiste após reboot
        if ! grep -q "$SWAPFILE" /etc/fstab; then
            echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab > /dev/null
        fi
        info "Swap configurado com sucesso."
    else
        warn "Swap ignorado. Monitore o uso de memória do Sidekiq manualmente."
    fi
else
    info "Swap já configurado em $SWAPFILE."
fi

# ---------------------------------------------------------------------------
# 3. Pull das imagens Docker
# ---------------------------------------------------------------------------
info "Baixando imagens Docker..."
docker compose pull

# ---------------------------------------------------------------------------
# 4. Inicializar / migrar o banco de dados
#
# O comando 'db:chatwoot_prepare' é idempotente:
#   - Se o banco for novo, cria o schema completo (db:schema:load) e
#     popula dados iniciais (db:seed).
#   - Se o banco já existir, aplica apenas as migrações pendentes.
# ---------------------------------------------------------------------------
info "Inicializando/migrando o banco de dados..."
docker compose run --rm \
    -e RAILS_ENV=production \
    chatwoot \
    bundle exec rails db:chatwoot_prepare

info "Banco de dados pronto."

# ---------------------------------------------------------------------------
# 5. Subir todos os serviços em modo daemon
# ---------------------------------------------------------------------------
info "Subindo os serviços em background..."
docker compose up -d

# ---------------------------------------------------------------------------
# 6. Relatório final
# ---------------------------------------------------------------------------
echo ""
info "============================================================"
info " Chatwoot Community Edition está no ar!"
info "============================================================"
info " Acesse:        a URL definida em FRONTEND_URL no seu arquivo .env"
info " Logs:          docker compose logs -f"
info " Status:        docker compose ps"
info " Parar tudo:    docker compose down"
info " Backup DB:     docker compose exec postgres pg_dump -U chatwoot chatwoot_production > backup.sql"
info ""
warn " Lembre-se de configurar o Cloudflare Tunnel e apontar o"
warn " webhook da WhatsApp Cloud API para:"
warn "   https://<seu-dominio>/webhooks/whatsapp"
info "============================================================"
