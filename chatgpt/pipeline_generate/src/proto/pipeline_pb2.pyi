from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Message(_message.Message):
    __slots__ = ["content", "params"]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    content: str
    params: Params
    def __init__(self, content: _Optional[str] = ..., params: _Optional[_Union[Params, _Mapping]] = ...) -> None: ...

class Answer(_message.Message):
    __slots__ = ["error", "content"]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    error: str
    content: str
    def __init__(self, error: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class Params(_message.Message):
    __slots__ = ["model", "temperature"]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    model: str
    temperature: float
    def __init__(self, model: _Optional[str] = ..., temperature: _Optional[float] = ...) -> None: ...
