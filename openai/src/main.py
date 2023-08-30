#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/10 16:03
# @Author  : Jack
# @File    : main.py
# @Software: PyCharm

import logging
import os
import sys
from concurrent import futures

import grpc

from proto import openai_pb2, openai_pb2_grpc
from utils import official

from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROXY = os.getenv("OPENAI_PROXY")
CUSTOM_BASE_PROMPT = os.getenv("CUSTOM_BASE_PROMPT")
GPT_ENGINE = os.getenv("GPT_ENGINE")

chatbot = official.Chatbot(
    api_key=OPENAI_API_KEY,
    org_id=OPENAI_ORG_ID,
    proxy=OPENAI_PROXY,
    custom_base_prompt=CUSTOM_BASE_PROMPT,
    engine=GPT_ENGINE
)


class ChatgptService(openai_pb2_grpc.ChatBotServicer):
    def SendPrompt(self, request, context) -> openai_pb2.ChatResponse:
        """
        send prompt and handle completion
        """
        try:
            answer = chatbot.ask(
                user_request=request.prompt,
                conversation_id=request.conversationId,
            )
        except Exception as e:
            print(f'ChatGPT API returned response failed: {e}')
            return openai_pb2.ChatResponse(code=0, msg=f'ChatGPT API returned response failed: {e}')
        else:
            return openai_pb2.ChatResponse(code=1, ret=answer)


def serve() -> None:
    """
    Run grpc service
    """
    port = 50051

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    openai_pb2_grpc.add_ChatBotServicer_to_server(ChatgptService(), server=server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Server started, listening on {port}")
    server.wait_for_termination()


if __name__ == '__main__':
    try:
        logging.basicConfig()
        serve()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit()
