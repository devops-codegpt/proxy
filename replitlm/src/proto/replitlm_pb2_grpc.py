# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from proto import replitlm_pb2 as replitlm__pb2


class CodeGeneratorStub(object):
    """The CodeGenerator service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendPrompt = channel.unary_unary(
            '/replitlm.CodeGenerator/SendPrompt',
            request_serializer=replitlm__pb2.CodeRequest.SerializeToString,
            response_deserializer=replitlm__pb2.CodeResponse.FromString,
        )


class CodeGeneratorServicer(object):
    """The CodeGenerator service definition.
    """

    def SendPrompt(self, request, context):
        """Send prompt to ReplitLM
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CodeGeneratorServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'SendPrompt': grpc.unary_unary_rpc_method_handler(
            servicer.SendPrompt,
            request_deserializer=replitlm__pb2.CodeRequest.FromString,
            response_serializer=replitlm__pb2.CodeResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler('replitlm.CodeGenerator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class CodeGenerator(object):
    """The CodeGenerator service definition.
    """
    @staticmethod
    def SendPrompt(
            request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/replitlm.CodeGenerator/SendPrompt',
            replitlm__pb2.CodeRequest.SerializeToString,
            replitlm__pb2.CodeResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata
        )
