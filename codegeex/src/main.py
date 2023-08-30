#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import argparse
import grpc
from concurrent import futures
from proto import codegeex_pb2, codegeex_pb2_grpc
# # pytorch
# from inference.inference import add_code_generation_args
# from inference.inference import predict
# from inference.inference import load
# oneflow
from inference.inference_oneflow import add_code_generation_args
from inference.inference_oneflow import predict
from inference.inference_oneflow import load


class CodeGeeXService(codegeex_pb2_grpc.CodeGeneratorServicer):
    def __init__(self, args):
        self.args = args
        self.seed = 1234

    def SendPrompt(self, request, context) -> codegeex_pb2.CodeResponse:
        """
        send prompt and handle completion
        """
        try:
            output = predict(self.args, request.prompt, request.lang, seed=self.seed)
        except Exception as e:
            print(f'CodeGeeX Inference returned response failed: {e}')
            return codegeex_pb2.CodeResponse(code=1, ret=None, msg=f'CodeGeeX Inference returned response failed: {e}')
        else:
            result = codegeex_pb2.CodeResponse()
            result.code = 0
            result.msg = "success"
            result.ret.code_list.extend(output["Code"])
            result.ret.completion_token_num = output["CompletionTokenNum"]
            result.ret.prompt_token_num = output["PromptTokenNum"]
            return result


def serve(args) -> None:
    """
        Run grpc service
        """
    port = 65091
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    codegeex_pb2_grpc.add_CodeGeneratorServicer_to_server(CodeGeeXService(args), server=server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Server started, listening on {port}")
    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()

    try:
        load(args)
    except Exception as e:
        print(f"failed to load CodeGeeX model, err: {e}")
        return

    serve(args)


if __name__ == "__main__":
    with torch.no_grad():
        main()
