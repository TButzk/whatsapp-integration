# whatsapp-integration
Integração com WhatsApp para centralização de atendimentos no Chatwoot, com respostas automáticas via Ollama rodando no host Windows.

## Resposta automática com Gemma3

Esta stack foi preparada para o seguinte desenho:

- Chatwoot, PostgreSQL, Redis, Nginx e Cloudflared em Docker
- Ollama rodando nativamente no Windows
- Bridge leve de auto-reply rodando no Windows
- Webhook do Chatwoot apontando para `/auto-reply/webhook`

Isso evita gastar mais RAM do Docker Desktop e mantém o modelo fora dos containers.

## Como subir o bridge de auto-reply

1. Crie um ambiente virtual Python no Windows.
2. Instale as dependências do diretório `auto_reply_bridge`.
3. Copie `auto_reply_bridge/.env.example` para `auto_reply_bridge/.env` e preencha os valores.
4. Inicie o Ollama no Windows.
5. Execute o bridge.

Exemplo no PowerShell:

```powershell
cd auto_reply_bridge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

## Variáveis do bridge

- `CHATWOOT_BASE_URL`: URL do Chatwoot. No host Windows, pode usar `http://localhost:65271`.
- `CHATWOOT_API_TOKEN`: token de API de um agente com permissão para responder conversas.
- `CHATWOOT_WEBHOOK_SECRET`: segredo do webhook configurado no Chatwoot.
- `OLLAMA_BASE_URL`: normalmente `http://localhost:11434`.
- `OLLAMA_MODEL`: modelo carregado no Ollama, por exemplo `gemma3`.
- `OLLAMA_SYSTEM_PROMPT`: instrução base do atendente virtual.

## Configuração no Chatwoot

1. Acesse `Settings -> Integrations -> Webhooks`.
2. Crie um webhook para o evento `message_created`.
3. Use a URL pública `https://SEU-DOMINIO/auto-reply/webhook`.
4. Guarde o secret do webhook e copie para `CHATWOOT_WEBHOOK_SECRET`.
5. Gere um `api_access_token` de um agente e copie para `CHATWOOT_API_TOKEN`.

## Regras do auto-reply

O bridge responde apenas quando:

- o evento é `message_created`
- a mensagem é `incoming`
- a mensagem não é privada
- existe texto para responder

Ele ignora mensagens enviadas pelo próprio agente para não criar loop.

## Healthcheck

Com o bridge rodando no Windows, valide:

```powershell
Invoke-WebRequest http://localhost:8000/healthz
```

E pela rota pública proxied pelo Nginx:

```powershell
Invoke-WebRequest https://SEU-DOMINIO/auto-reply/healthz
```
