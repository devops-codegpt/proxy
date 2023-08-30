#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/24 11:39
# @Author  : Jack
# @File    : main.py
# @Software: PyCharm
import asyncio
import logging
import os
import pathlib
import platform
import socket
import sys

import consul
import grpc
import langchain
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from proto import pipeline_pb2_grpc, pipeline_pb2
from callback import StreamingLLMCallbackHandler

plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

default_port = 8097
service_name = "pipeline"


# Fill your openai api key.
os.environ["OPENAI_API_KEY"] = ""


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
        name=service_name,
        address=local_ip,
        port=srv_port,
        service_id=f"{service_name}-{local_ip}",
        timeout=10
    )
    return client


def init_pipeline() -> langchain.LLMChain:
    template = """You are a senior devops expert.
The following is an example of the pipeline primitive yaml content of an application: {example}.
You understand the basic syntax of the primitive based on examples. The user will provide his natural language description, Understand the description and only return the corresponding yaml file of the description and nothing more. If you can't extract the information to generate yaml content, just say that you don't know."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(
        llm=ChatOpenAI(streaming=True, verbose=True),
        prompt=chat_prompt,
        callbacks=[],
        verbose=True
    )
    return chain


class ChatgptService(pipeline_pb2_grpc.ChatgptServicer):
    def __init__(self, chain, example: str):
        self.chain = chain
        self.example = example

    async def Send(self, request: pipeline_pb2.Message, context: grpc.aio.ServicerContext):
        print(request.content)
        stream_handler = StreamingLLMCallbackHandler(context, pipeline_pb2)
        await self.chain.acall(
            {
                "example": self.example,
                "text": request.content,
            },
            callbacks=[stream_handler]
        )


async def serve() -> None:
    """
    Run grpc service
    """
    with open(r"example.yaml", "r", encoding="utf-8") as f:
        content = f.read()

    server = grpc.aio.server()
    chain = init_pipeline()
    pipeline_pb2_grpc.add_ChatgptServicer_to_server(ChatgptService(chain, content), server=server)
    server.add_insecure_port(f"[::]:{default_port}")
    await server.start()
    print(f"Server started, listening on {default_port}")
    await server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Fill your consul service address and port.
    client = register_service("127.0.0.1", 8500, default_port)
    try:
        asyncio.get_event_loop().run_until_complete(serve())
    except KeyboardInterrupt:
        client.agent.service.deregister(f"{service_name}-{get_host_ip()}")
        print("\nExiting...")
        sys.exit()
