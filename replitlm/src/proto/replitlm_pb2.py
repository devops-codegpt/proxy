# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: replitlm.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0ereplitlm.proto\x12\x08replitlm\"+\n\x0b\x43odeRequest\x12\x0e\n\x06prompt\x18\x01 \x01(\t\x12\x0c\n\x04lang\x18\x02 \x01(\t\"H\n\x0c\x43odeResponse\x12\x1d\n\x03ret\x18\x01 \x01(\x0b\x32\x10.replitlm.Result\x12\x0c\n\x04\x63ode\x18\x02 \x01(\x05\x12\x0b\n\x03msg\x18\x03 \x01(\t\"S\n\x06Result\x12\x11\n\tcode_list\x18\x01 \x03(\t\x12\x1c\n\x14\x63ompletion_token_num\x18\x02 \x01(\x05\x12\x18\n\x10prompt_token_num\x18\x03 \x01(\x05\x32N\n\rCodeGenerator\x12=\n\nSendPrompt\x12\x15.replitlm.CodeRequest\x1a\x16.replitlm.CodeResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'replitlm_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._options = None
  _globals['_CODEREQUEST']._serialized_start = 28
  _globals['_CODEREQUEST']._serialized_end = 71
  _globals['_CODERESPONSE']._serialized_start = 73
  _globals['_CODERESPONSE']._serialized_end = 145
  _globals['_RESULT']._serialized_start = 147
  _globals['_RESULT']._serialized_end = 230
  _globals['_CODEGENERATOR']._serialized_start = 232
  _globals['_CODEGENERATOR']._serialized_end = 310
# @@protoc_insertion_point(module_scope)