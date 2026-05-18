from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class DependencyStatus(BaseModel):
    configured: bool
    detail: str


class ReadyResponse(BaseModel):
    status: str
    service: str
    dependencies: dict[str, DependencyStatus]


class ChatwootContact(BaseModel):
    id: int | None = None
    name: str | None = None
    phone_number: str | None = None
    email: str | None = None


class ChatwootConversation(BaseModel):
    id: int
    status: str | None = None
    channel: str | None = None


class ChatwootAccount(BaseModel):
    id: int | None = None


class ChatwootMessage(BaseModel):
    id: int | None = None
    content: str = Field(default="")
    message_type: str | int | None = None
    private: bool = False


class ChatwootWebhookPayload(BaseModel):
    event: str = Field(default="message_created")
    account_id: int | None = None
    account: ChatwootAccount | None = None
    content: str = Field(default="")
    message_type: str | int | None = None
    private: bool = False
    conversation: ChatwootConversation
    contact: ChatwootContact | None = None
    message: ChatwootMessage | None = None
    sender: dict[str, Any] | None = None


class WhatsAppWebhookPayload(BaseModel):
    message_id: str
    from_number: str
    content: str = Field(default="")
    profile_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InternalProcessMessageRequest(BaseModel):
    channel: str
    external_message_id: str
    conversation_id: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdminKnowledgeImportRequest(BaseModel):
    source_paths: list[str] = Field(default_factory=list)
    reindex: bool = False


class QAPair(BaseModel):
    id: str
    question: str
    answer: str
    category: str = "geral"
    aliases: list[str] = Field(default_factory=list)


class ImportQARequest(BaseModel):
    pairs: list[QAPair]


class ImportQAResponse(BaseModel):
    imported: int
    collection: str


class QAPairListItem(BaseModel):
    qa_id: str
    question: str
    answer: str
    category: str


class QAPairListResponse(BaseModel):
    total: int
    pairs: list[QAPairListItem]


class AcceptedResponse(BaseModel):
    accepted: bool = True
    channel: str
    reason: str
    message_id: str | None = None


class LlmStatusResponse(BaseModel):
    base_url: str
    fast_model: str
    smart_model: str
    timeout_seconds: int
    max_tokens: int
    temperature: float


class ProcessingResponse(BaseModel):
    intent: str
    confidence: float
    action: str
    reason: str
    customer_message: str
    audit_reasons: list[str] = Field(default_factory=list)


class ShopifyCustomerOAuthAuthorizeResponse(BaseModel):
    authorization_url: str
    state: str
    redirect_uri: str
    code_verifier: str | None = None
    uses_pkce: bool = True


class ShopifyCustomerOAuthExchangeRequest(BaseModel):
    code: str
    redirect_uri: str | None = None
    code_verifier: str | None = None


class ShopifyCustomerOAuthExchangeResponse(BaseModel):
    ok: bool
    token_type: str | None = None
    expires_in: int | None = None
    scope: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None