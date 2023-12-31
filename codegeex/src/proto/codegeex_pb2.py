# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: codegeex.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='codegeex.proto',
    package='codegeex',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0e\x63odegeex.proto\x12\x08\x63odegeex\"+\n\x0b\x43odeRequest\x12\x0e\n\x06prompt\x18\x01 \x01(\t\x12\x0c\n\x04lang\x18\x02 \x01(\t\"H\n\x0c\x43odeResponse\x12\x1d\n\x03ret\x18\x01 \x01(\x0b\x32\x10.codegeex.Result\x12\x0c\n\x04\x63ode\x18\x02 \x01(\x05\x12\x0b\n\x03msg\x18\x03 \x01(\t\"S\n\x06Result\x12\x11\n\tcode_list\x18\x01 \x03(\t\x12\x1c\n\x14\x63ompletion_token_num\x18\x02 \x01(\x05\x12\x18\n\x10prompt_token_num\x18\x03 \x01(\x05\x32N\n\rCodeGenerator\x12=\n\nSendPrompt\x12\x15.codegeex.CodeRequest\x1a\x16.codegeex.CodeResponse\"\x00\x62\x06proto3'
)


_CODEREQUEST = _descriptor.Descriptor(
    name='CodeRequest',
    full_name='codegeex.CodeRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='prompt', full_name='codegeex.CodeRequest.prompt', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='lang', full_name='codegeex.CodeRequest.lang', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=28,
    serialized_end=71,
)


_CODERESPONSE = _descriptor.Descriptor(
    name='CodeResponse',
    full_name='codegeex.CodeResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='ret', full_name='codegeex.CodeResponse.ret', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='code', full_name='codegeex.CodeResponse.code', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='msg', full_name='codegeex.CodeResponse.msg', index=2,
            number=3, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=73,
    serialized_end=145,
)


_RESULT = _descriptor.Descriptor(
    name='Result',
    full_name='codegeex.Result',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='code_list', full_name='codegeex.Result.code_list', index=0,
            number=1, type=9, cpp_type=9, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='completion_token_num', full_name='codegeex.Result.completion_token_num', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='prompt_token_num', full_name='codegeex.Result.prompt_token_num', index=2,
            number=3, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=147,
    serialized_end=230,
)

_CODERESPONSE.fields_by_name['ret'].message_type = _RESULT
DESCRIPTOR.message_types_by_name['CodeRequest'] = _CODEREQUEST
DESCRIPTOR.message_types_by_name['CodeResponse'] = _CODERESPONSE
DESCRIPTOR.message_types_by_name['Result'] = _RESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CodeRequest = _reflection.GeneratedProtocolMessageType('CodeRequest', (_message.Message,), {
    'DESCRIPTOR': _CODEREQUEST,
    '__module__': 'codegeex_pb2'
    # @@protoc_insertion_point(class_scope:codegeex.CodeRequest)
})
_sym_db.RegisterMessage(CodeRequest)

CodeResponse = _reflection.GeneratedProtocolMessageType('CodeResponse', (_message.Message,), {
    'DESCRIPTOR': _CODERESPONSE,
    '__module__': 'codegeex_pb2'
    # @@protoc_insertion_point(class_scope:codegeex.CodeResponse)
})
_sym_db.RegisterMessage(CodeResponse)

Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), {
    'DESCRIPTOR': _RESULT,
    '__module__': 'codegeex_pb2'
    # @@protoc_insertion_point(class_scope:codegeex.Result)
})
_sym_db.RegisterMessage(Result)


_CODEGENERATOR = _descriptor.ServiceDescriptor(
    name='CodeGenerator',
    full_name='codegeex.CodeGenerator',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=232,
    serialized_end=310,
    methods=[
        _descriptor.MethodDescriptor(
            name='SendPrompt',
            full_name='codegeex.CodeGenerator.SendPrompt',
            index=0,
            containing_service=None,
            input_type=_CODEREQUEST,
            output_type=_CODERESPONSE,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ]
)
_sym_db.RegisterServiceDescriptor(_CODEGENERATOR)
DESCRIPTOR.services_by_name['CodeGenerator'] = _CODEGENERATOR
# @@protoc_insertion_point(module_scope)
