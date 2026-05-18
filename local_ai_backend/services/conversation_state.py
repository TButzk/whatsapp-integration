"""In-memory conversation state store with TTL.

Keeps track of the last pending intent per conversation so that
multi-turn replies (e.g. user responds with just a CPF after the bot
asked for an identifier) are resolved correctly.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


_DEFAULT_TTL_SECONDS = 600  # 10 minutes of inactivity clears the state


@dataclass
class ConversationState:
    conversation_id: str
    # Intent the bot was trying to resolve in the previous turn
    pending_intent: Optional[str] = None
    # Whether the bot already asked the user for an order identifier
    awaiting_identifier: bool = False
    # Current position in the menu-driven conversation flow
    flow_state: str = "START"
    last_updated: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_updated = time.monotonic()

    def is_expired(self, ttl: float = _DEFAULT_TTL_SECONDS) -> bool:
        return (time.monotonic() - self.last_updated) > ttl


class ConversationStateStore:
    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._store: dict[str, ConversationState] = {}
        self._ttl = ttl_seconds

    def get(self, conversation_id: str) -> ConversationState:
        state = self._store.get(conversation_id)
        if state is None or state.is_expired(self._ttl):
            state = ConversationState(conversation_id=conversation_id)
            self._store[conversation_id] = state
        return state

    def save(self, state: ConversationState) -> None:
        state.touch()
        self._store[state.conversation_id] = state

    def clear(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)
