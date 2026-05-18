from __future__ import annotations

from dataclasses import dataclass
import re

from local_ai_backend.intent import extract_order_number
from local_ai_backend.schemas import ChatwootContact

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?\(?\d{2}\)?[\s-]?\d{4,5}[\s-]?\d{4}\b")
_CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")


@dataclass(slots=True)
class CustomerIdentity:
    email: str | None = None
    phone: str | None = None
    cpf: str | None = None
    order_number: str | None = None

    def has_identifier(self) -> bool:
        return bool(self.order_number or self.email or self.phone or self.cpf)


def _normalize_email(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _normalize_phone(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None

    has_plus = raw.startswith("+")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return f"+{digits}" if has_plus else digits


def _normalize_cpf(value: str | None) -> str | None:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) != 11:
        return None
    return digits


def extract_customer_identity(content: str, contact: ChatwootContact | None = None) -> CustomerIdentity:
    text = content or ""

    contact_email = _normalize_email(contact.email) if contact else None
    contact_phone = _normalize_phone(contact.phone_number) if contact else None

    message_email = None
    message_phone = None
    message_cpf = None

    email_match = _EMAIL_RE.search(text)
    if email_match:
        message_email = _normalize_email(email_match.group(0))

    phone_match = _PHONE_RE.search(text)
    if phone_match:
        message_phone = _normalize_phone(phone_match.group(0))

    cpf_match = _CPF_RE.search(text)
    if cpf_match:
        message_cpf = _normalize_cpf(cpf_match.group(0))

    return CustomerIdentity(
        email=contact_email or message_email,
        phone=contact_phone or message_phone,
        cpf=message_cpf,
        order_number=extract_order_number(text),
    )
