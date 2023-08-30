#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/26 14:04
# @Author  : Jack
# @File    : main.py
# @Software: PyCharm

import asyncio
import logging
import platform
import socket
import sys

import consul
import os
import grpc
from langchain import PromptTemplate, FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from proto import chatgpt_pb2_grpc, chatgpt_pb2

from callback import StreamingLLMCallbackHandler

import pathlib


# Fill your openai api key.
os.environ["OPENAI_API_KEY"] = ""

plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

default_port = 8098


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
        name="doc_qa",
        address=local_ip,
        port=srv_port,
        service_id=f"doc_qa-{local_ip}",
        timeout=10
    )
    return client


def init_qa():
    # Fill in the directory address of your vector database
    vector_store = FAISS.load_local("vector_store", embeddings=OpenAIEmbeddings())
    doc_retriever = vector_store.as_retriever()
    # Create qa chain
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Answer in Chinese:"""
    qa_prompt_template = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa_llm = ChatOpenAI(streaming=True, verbose=True, temperature=0)
    qa_chain = load_qa_chain(qa_llm, chain_type="stuff", verbose=True, prompt=qa_prompt_template)
    doc_qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=doc_retriever)
    return doc_qa


class ChatgptService(chatgpt_pb2_grpc.ChatgptServicer):
    def __init__(self, qa_client):
        self.qa_client = qa_client

    async def Send(self, request: chatgpt_pb2.Message, context: grpc.aio.ServicerContext):
        stream_handler = StreamingLLMCallbackHandler(context, chatgpt_pb2)
        await self.qa_client.acall(
            {"query": request.content},
            callbacks=[stream_handler]
        )


async def serve() -> None:
    """
    Run grpc service
    """
    server = grpc.aio.server()
    qa_client = init_qa()
    chatgpt_pb2_grpc.add_ChatgptServicer_to_server(ChatgptService(qa_client), server=server)
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
        client.agent.service.deregister(f"doc_qa-{get_host_ip()}")
        print("\nExiting...")
        sys.exit()
