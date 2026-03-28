# whatsapp-integration
Integração com WhatsApp para centralização de atendimentos no Chatwoot, com respostas automáticas via Ollama rodando no host Windows.

## Resposta automática com Gemma3

Esta stack foi preparada para o seguinte desenho:

- Chatwoot, PostgreSQL, Redis, Nginx e Cloudflared em Docker
- Ollama rodando nativamente no Windows
- Bridge leve de auto-reply rodando no Windows
- Webhook do Chatwoot apontando para `/auto-reply/webhook`

Isso evita gastar mais RAM do Docker Desktop e mantém o modelo fora dos containers.

## Funcionalidades do bridge

| Capacidade | Descrição |
|---|---|
| **Histórico de conversa** | Busca as últimas mensagens da conversa para manter contexto entre turnos |
| **Shopify mock** | Responde consultas de pedidos e produtos sem precisar de API real |
| **RAG de documentos** | Recupera trechos relevantes de documentos da empresa para responder perguntas institucionais |
| **Fallback de modelo** | Se o modelo principal falhar, tenta automaticamente o modelo de fallback |

## Como subir o bridge de auto-reply

1. Use Python 3.11 (compatível com `chromadb==0.6.3`) no Windows.
2. Instale as dependências do diretório `auto_reply_bridge`.
3. Copie `auto_reply_bridge/.env.example` para `auto_reply_bridge/.env` e preencha os valores.
4. Inicie o Ollama no Windows.
5. Execute o bridge.

Exemplo no PowerShell:

```powershell
cd auto_reply_bridge
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

## Ingestão de documentos (RAG)

Coloque seus documentos (`.txt`, `.md`, `.html`, `.pdf`) na pasta `auto_reply_bridge/docs/` e execute:

```powershell
cd auto_reply_bridge
python -c "from rag import ingest_documents; ingest_documents()"
```

O pipeline de embeddings requer o modelo configurado em `RAG_EMBED_MODEL` rodando no Ollama:

```powershell
ollama pull nomic-embed-text
```

A ingestão pode ser re-executada a qualquer momento — os chunks são armazenados com `upsert`.

## Variáveis do bridge

### Chatwoot e aplicação

| Variável | Padrão | Descrição |
|---|---|---|
| `CHATWOOT_BASE_URL` | `http://localhost:65271` | URL do Chatwoot |
| `CHATWOOT_API_TOKEN` | — | Token de API de um agente |
| `CHATWOOT_WEBHOOK_SECRET` | — | Secret do webhook |
| `MAX_RESPONSE_CHARS` | `1200` | Tamanho máximo da resposta |
| `IGNORE_BOT_PREFIX` | `!botoff` | Prefixo para ignorar o bot |

### Ollama

| Variável | Padrão | Descrição |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do Ollama |
| `RAG_OLLAMA_BASE_URL` | `OLLAMA_BASE_URL` | URL do Ollama usada apenas para embeddings (RAG) |
| `OLLAMA_MODEL` | `gemma3` | Modelo principal |
| `OLLAMA_FALLBACK_MODEL` | `phi3:medium` | Modelo de fallback automático |
| `OLLAMA_SYSTEM_PROMPT` | *(ver .env.example)* | Instrução base do atendente |
| `OLLAMA_TIMEOUT_MAX_SECONDS` | `300` | Limite superior aplicado aos timeouts para evitar valores extremos |
| `OLLAMA_MAIN_TIMEOUT` | `90` | Timeout do modelo principal (segundos) |
| `OLLAMA_FALLBACK_TIMEOUT` | `60` | Timeout do fallback (segundos) |
| `OLLAMA_KEEP_ALIVE` | `5m` | Tempo que o modelo fica carregado após resposta |
| `OLLAMA_NUM_GPU` | `999` | Número de camadas para offload em GPU (NVIDIA) |
| `OLLAMA_UNAVAILABLE_MESSAGE` | `No momento estou com instabilidade para responder. Pode tentar novamente em instantes?` | Mensagem enviada quando principal e fallback falham |

### Histórico de conversa

| Variável | Padrão | Descrição |
|---|---|---|
| `CHAT_HISTORY_ENABLED` | `true` | Liga/desliga o histórico |
| `CHAT_HISTORY_MAX_MESSAGES` | `10` | Máximo de mensagens no contexto |
| `CHAT_HISTORY_MAX_CHARS` | `3000` | Máximo de caracteres de histórico |
| `CHAT_HISTORY_INCLUDE_AGENT` | `true` | Incluir mensagens do agente humano |

### Shopify

| Variável | Padrão | Descrição |
|---|---|---|
| `SHOPIFY_MODE` | `mock` | `mock` = dados locais; `real` = API Shopify |
| `SHOPIFY_STORE_NAME` | `Nome da Loja` | Nome exibido nas respostas |
| `SHOPIFY_CURRENCY` | `BRL` | Moeda |
| `SHOPIFY_MOCK_DELAY_MS` | `0` | Atraso simulado em ms |

### RAG de documentos

| Variável | Padrão | Descrição |
|---|---|---|
| `RAG_ENABLED` | `true` | Liga/desliga o RAG |
| `RAG_DOCS_PATH` | `./docs` | Pasta com os documentos |
| `RAG_VECTOR_DB_PATH` | `./data/vectorstore` | Pasta do banco vetorial |
| `RAG_TOP_K` | `4` | Chunks recuperados por consulta |
| `RAG_MAX_CONTEXT_CHARS` | `2500` | Máximo de caracteres de contexto RAG |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Modelo de embeddings |

Observação: se chat e embeddings estiverem em instâncias diferentes do Ollama, configure `RAG_OLLAMA_BASE_URL` apontando para a instância que possui o modelo `RAG_EMBED_MODEL`.

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

## Testes

```powershell
cd auto_reply_bridge
python -m pytest tests/ -v
```

## Exemplos de mensagens e respostas esperadas

| Mensagem do cliente | Intenção detectada | Resposta esperada |
|---|---|---|
| `Qual o status do pedido #1001?` | ORDER | Status + itens + previsão |
| `Meu pedido é #9999` | ORDER | Pedido não encontrado |
| `Vocês têm tênis?` | PRODUCT | Lista de até 3 tênis disponíveis |
| `Qual a política de troca?` | INSTITUTIONAL | Resposta baseada nos documentos da empresa |
| `e o prazo?` (após pergunta anterior) | GENERAL + histórico | Resposta contextualizando o histórico |
| `Oi, tudo bem?` | GENERAL | Resposta genérica do atendente |

## Próximos passos para integrar a API Shopify real

1. Crie um app privado na Shopify com permissão `read_orders` e `read_products`.
2. Adicione `SHOPIFY_ACCESS_TOKEN` e `SHOPIFY_STORE_URL` ao `.env`.
3. Crie `shopify_real.py` com as mesmas assinaturas de `shopify_mock.py` (`get_order_status`, `search_products`).
4. Em `app.py`, importe o módulo real quando `SHOPIFY_MODE != "mock"`.
5. Adicione tratamento de rate limit (429) com retry e backoff.
6. Nunca versione o `SHOPIFY_ACCESS_TOKEN` — use variáveis de ambiente ou um secrets manager.

## Estrutura dos arquivos do bridge

```
auto_reply_bridge/
├── app.py              # Orquestração principal (Flask + worker)
├── chat_history.py     # Fase 1: histórico de conversa
├── shopify_mock.py     # Fase 2: Shopify em modo mock
├── intent.py           # Fase 2: detecção de intenção
├── rag.py              # Fase 3: ingestão e busca RAG
├── docs/               # Documentos da empresa (txt, md, html, pdf)
├── data/vectorstore/   # Banco vetorial Chroma (gerado pelo ingest)
├── requirements.txt
├── .env.example
└── tests/
    └── test_bridge.py
```

