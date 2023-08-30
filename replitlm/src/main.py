import sys
import argparse
import logging
import asyncio
import socket
import consul
import grpc
from concurrent import futures
from proto import replitlm_pb2, replitlm_pb2_grpc

# import torch
# from inference.inference import add_code_generation_args
# from inference.inference import predict
# from inference.inference import load

import oneflow as torch
from inference.inference_oneflow import add_code_generation_args
from inference.inference_oneflow import predict
from inference.inference_oneflow import load


default_port = 65092


def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def register_service(consul_addr: str, consul_port: int, srv_port: int) -> consul.Consul:
    local_ip = get_host_ip()
    client = consul.Consul(host=consul_addr, port=consul_port, verify=False)
    client.agent.service.register(
        name="replitlm",
        address=local_ip,
        port=srv_port,
        service_id=f"replitlm-{local_ip}",
        timeout=10
    )
    return client


class ReplitLMService(replitlm_pb2_grpc.CodeGeneratorServicer):
    def __init__(self, args):
        self.args = args

    def SendPrompt(self, request, context) -> replitlm_pb2.CodeResponse:
        """
        send prompt and handle completion
        """
        try:
            output = predict(self.args, request.prompt, request.lang)
        except Exception as e:
            print(f'ReplitLM Inference returned response failed: {e}')
            return replitlm_pb2.CodeResponse(code=1, ret=None, msg=f'ReplitLM Inference returned response failed: {e}')
        else:
            result = replitlm_pb2.CodeResponse()
            result.code = 0
            result.msg = "success"
            result.ret.code_list.extend(output["Code"])
            result.ret.completion_token_num = output["CompletionTokenNum"]
            result.ret.prompt_token_num = output["PromptTokenNum"]
            return result


async def serve(args) -> None:
    """
    Run grpc service
    """
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    replitlm_pb2_grpc.add_CodeGeneratorServicer_to_server(ReplitLMService(args), server=server)
    server.add_insecure_port(f"[::]:{default_port}")
    await server.start()
    print(f"Server started, listening on {default_port}")
    await server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()

    try:
        load(args)
    except Exception as e:
        print(f"failed to load ReplitLM model, err: {e}")
        return -1

    logging.basicConfig(level=logging.INFO)
    client = register_service("192.168.1.20", 8500, default_port)
    try:
        asyncio.get_event_loop().run_until_complete(serve(args))
    except KeyboardInterrupt:
        client.agent.service.deregister(f"replitlm-{get_host_ip()}")
        print("\nExiting...")
        return 0


if __name__ == "__main__":
    with torch.no_grad():
        ret = main()
    sys.exit(ret)
