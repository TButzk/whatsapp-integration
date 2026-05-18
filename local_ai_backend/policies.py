from dataclasses import dataclass, field
from enum import Enum

from local_ai_backend.intent import Intent, IntentResult


class DecisionAction(str, Enum):
    RESPOND = "respond"
    ASK_FOR_MORE_INFO = "ask_for_more_info"
    HANDOFF_HUMAN = "handoff_human"


@dataclass(slots=True)
class Decision:
    action: DecisionAction
    reason: str
    customer_message: str
    audit_reasons: list[str] = field(default_factory=list)


def decide_next_action(intent_result: IntentResult) -> Decision:
    audit_reasons = list(intent_result.reasons)

    if intent_result.intent == Intent.FALAR_COM_HUMANO:
        return Decision(
            action=DecisionAction.HANDOFF_HUMAN,
            reason="customer_requested_human",
            customer_message="Vou encaminhar seu atendimento para um especialista humano.",
            audit_reasons=audit_reasons,
        )

    if intent_result.intent == Intent.RECLAMACAO:
        return Decision(
            action=DecisionAction.HANDOFF_HUMAN,
            reason="complaint_requires_human_review",
            customer_message="Sinto muito pelo problema. Vou encaminhar seu caso para atendimento humano agora.",
            audit_reasons=audit_reasons,
        )

    if intent_result.intent in {Intent.PEDIDO_STATUS, Intent.TRACKING}:
        if intent_result.order_number or intent_result.has_customer_identifier:
            return Decision(
                action=DecisionAction.RESPOND,
                reason="order_lookup_ready",
                customer_message="Entendi. Vou consultar as informações do seu pedido.",
                audit_reasons=audit_reasons,
            )
        return Decision(
            action=DecisionAction.ASK_FOR_MORE_INFO,
            reason="missing_order_identifier",
            customer_message=(
                "Para consultar pedido ou rastreio, envie o número do pedido ou um dado de identificação, "
                "como e-mail, telefone ou CPF/CNPJ."
            ),
            audit_reasons=audit_reasons,
        )

    if intent_result.intent == Intent.ATACADO:
        return Decision(
            action=DecisionAction.HANDOFF_HUMAN,
            reason="wholesale_requires_sales_flow",
            customer_message=(
                "Posso encaminhar seu atendimento para o time comercial de atacado. "
                "Se quiser, já envie empresa, CNPJ e volume aproximado."
            ),
            audit_reasons=audit_reasons,
        )

    if intent_result.confidence < 0.55 or intent_result.intent == Intent.DESCONHECIDO:
        return Decision(
            action=DecisionAction.HANDOFF_HUMAN,
            reason="low_confidence",
            customer_message="Não tenho segurança para responder isso agora. Vou encaminhar para atendimento humano.",
            audit_reasons=audit_reasons,
        )

    return Decision(
        action=DecisionAction.RESPOND,
        reason="policy_allows_response",
        customer_message="Entendi. Vou seguir com a resposta usando apenas os dados confirmados.",
        audit_reasons=audit_reasons,
    )