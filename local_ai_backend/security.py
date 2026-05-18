import hashlib
import hmac


def verify_hmac_signature(body: bytes, provided_signature: str | None, secret: str) -> bool:
    if not secret:
        return True
    if not provided_signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    normalized_signature = provided_signature.strip()
    if normalized_signature.startswith("sha256="):
        normalized_signature = normalized_signature.split("=", 1)[1]
    return hmac.compare_digest(expected, normalized_signature)