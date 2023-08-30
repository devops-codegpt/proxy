#!/usr/bin/python
# -*- coding: utf-8 -*-
import grpc
from proto import replitlm_pb2, replitlm_pb2_grpc


def main():
    with grpc.insecure_channel("0.0.0.0:65092") as channel:
        stub = replitlm_pb2_grpc.CodeGeneratorStub(channel)
        request = replitlm_pb2.CodeRequest(prompt="// Write a function that returns the sum of the numbers from 1 to n.\n// For example, if n is 5, then the function should return 1 + 2 + 3 + 4 + 5.\n\npublic class SumOfNumbers {", lang="java")
        response = stub.SendPrompt(request)
    print("response:", response)


if __name__ == "__main__":
    main()
