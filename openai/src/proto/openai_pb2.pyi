from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ChatRequest(_message.Message):
    __slots__ = ["conversationId", "prompt"]
    CONVERSATIONID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    conversationId: str
    prompt: str
    def __init__(self, prompt: _Optional[str] = ..., conversationId: _Optional[str] = ...) -> None: ...

class ChatResponse(_message.Message):
    __slots__ = ["code", "msg", "ret"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    MSG_FIELD_NUMBER: _ClassVar[int]
    RET_FIELD_NUMBER: _ClassVar[int]
    code: int
    msg: str
    ret: str
    def __init__(self, ret: _Optional[str] = ..., code: _Optional[int] = ..., msg: _Optional[str] = ...) -> None: ...
