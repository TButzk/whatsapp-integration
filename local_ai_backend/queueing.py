import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import redis


@dataclass(slots=True)
class QueueMessage:
    channel: str
    external_message_id: str
    conversation_id: str | None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MessagePublisher(Protocol):
    def publish(self, message: QueueMessage) -> str: ...


class InMemoryMessagePublisher:
    def __init__(self) -> None:
        self.messages: list[QueueMessage] = []

    def publish(self, message: QueueMessage) -> str:
        self.messages.append(message)
        return message.external_message_id


class RedisMessagePublisher:
    def __init__(self, redis_url: str, queue_name: str) -> None:
        self._queue_name = queue_name
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def publish(self, message: QueueMessage) -> str:
        payload = json.dumps(asdict(message), ensure_ascii=False)
        self._client.rpush(self._queue_name, payload)
        return message.external_message_id