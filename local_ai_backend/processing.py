from pydantic import BaseModel, Field

from local_ai_backend.intent import Intent, IntentResult, classify_intent, has_customer_identifier
from local_ai_backend.policies import DecisionAction, decide_next_action
from local_ai_backend.services.conversation_state import ConversationState


class ProcessingResult(BaseModel):
    intent: Intent
    confidence: float
    action: DecisionAction
    reason: str
    customer_message: str
    audit_reasons: list[str] = Field(default_factory=list)


def _resolve_intent_with_context(content: str, state: ConversationState | None) -> IntentResult:
    """Classify intent, using conversation state to handle multi-turn flows.

    If the bot previously asked for an order identifier (awaiting_identifier=True)
    and the user replies with just a CPF/email/phone/order number, we restore the
    pending intent instead of classifying the short reply as DESCONHECIDO.
    """
    result = classify_intent(content)

    if (
        state is not None
        and state.awaiting_identifier
        and state.pending_intent is not None
        and result.intent == Intent.DESCONHECIDO
        and (result.has_customer_identifier or result.order_number)
    ):
        try:
            restored_intent = Intent(state.pending_intent)
        except ValueError:
            return result
        return IntentResult(
            intent=restored_intent,
            confidence=0.90,
            reasons=[*result.reasons, f"context_restored:{state.pending_intent}"],
            order_number=result.order_number,
            has_customer_identifier=result.has_customer_identifier,
        )

    return result


def process_message(content: str, state: ConversationState | None = None) -> ProcessingResult:
    intent_result = _resolve_intent_with_context(content, state)
    decision = decide_next_action(intent_result)

    # Update state so next turn knows what we are waiting for
    if state is not None:
        if decision.action == DecisionAction.ASK_FOR_MORE_INFO and decision.reason == "missing_order_identifier":
            state.awaiting_identifier = True
            state.pending_intent = intent_result.intent.value
        else:
            state.awaiting_identifier = False
            state.pending_intent = None

    return ProcessingResult(
        intent=intent_result.intent,
        confidence=intent_result.confidence,
        action=decision.action,
        reason=decision.reason,
        customer_message=decision.customer_message,
        audit_reasons=intent_result.reasons,
    )