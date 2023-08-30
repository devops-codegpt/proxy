#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 16:23
# @Author  : Jack
# @File    : main.py
# @Software: PyCharm
import asyncio
import logging
import socket
import sys

import consul
import langchain
import os
import grpc
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from proto import chatgpt_pb2_grpc, chatgpt_pb2

from callback import StreamingLLMCallbackHandler

default_port = 8099


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
        name="chatgpt",
        address=local_ip,
        port=srv_port,
        service_id=f"chatgpt-{local_ip}",
        timeout=10
    )
    return client


def init_chatgpt() -> langchain.LLMChain:
    llm = ChatOpenAI(streaming=True, verbose=True, temperature=0.6)
    # Get prompt template
    template = ("""Assistant is a large language model trained by OpenAI.
    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
        {human_input}
        Assistant:""")
    chat_prompt = PromptTemplate(
        input_variables=["human_input"],
        template=template
    )
    # Construct Chain
    chain = LLMChain(llm=llm, prompt=chat_prompt, callbacks=[], verbose=True)
    return chain

# Fill your openai api key.
os.environ["OPENAI_API_KEY"] = ""


class ChatgptService(chatgpt_pb2_grpc.ChatgptServicer):
    def __init__(self, chain):
        self.chain = chain

    async def Send(self, request: chatgpt_pb2.Message, context: grpc.aio.ServicerContext):
        stream_handler = StreamingLLMCallbackHandler(context, chatgpt_pb2)
        await self.chain.acall(
            {"human_input": request.content},
            callbacks=[stream_handler]
        )


async def serve() -> None:
    """
    Run grpc service
    """
    server = grpc.aio.server()
    chain = init_chatgpt()
    chatgpt_pb2_grpc.add_ChatgptServicer_to_server(ChatgptService(chain), server=server)
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
        client.agent.service.deregister(f"chatgpt-{get_host_ip()}")
        print("\nExiting...")
        sys.exit()
