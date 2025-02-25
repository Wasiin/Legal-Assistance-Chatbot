from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    type: str
    id: str
    quotedMessageId: Optional[str] = None
    quoteToken: Optional[str] = None
    text: str

class DeliveryContext(BaseModel):
    isRedelivery: bool

class Source(BaseModel):
    type: str
    groupId: Optional[str] = None
    userId: Optional[str] = None

class Event(BaseModel):
    type: str
    message: Message
    webhookEventId: str
    deliveryContext: DeliveryContext
    timestamp: int
    source: Source
    replyToken: str
    mode: str

class ReceiveLine(BaseModel):
    destination: str
    events: List[Event]