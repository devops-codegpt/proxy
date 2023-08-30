from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Answer(_message.Message):
    __slots__ = ["content", "error"]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    content: str
    error: str
    def __init__(self, error: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class Message(_message.Message):
    __slots__ = ["content", "params"]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    content: str
    params: Params
    def __init__(self, content: _Optional[str] = ..., params: _Optional[_Union[Params, _Mapping]] = ...) -> None: ...

class Params(_message.Message):
    __slots__ = ["model", "temperature"]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    model: str
    temperature: float
    def __init__(self, model: _Optional[str] = ..., temperature: _Optional[float] = ...) -> None: ...
