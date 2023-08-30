#!/usr/bin/python
# -*- coding: utf-8 -*-

import grpc
from proto import codegeex_pb2, codegeex_pb2_grpc


def main():
    with grpc.insecure_channel("0.0.0.0:50051") as channel:
        stub = codegeex_pb2_grpc.CodeGeneratorStub(channel)
        request = codegeex_pb2.CodeRequest(prompt="// Write a function that returns the sum of the numbers from 1 to n.\n// For example, if n is 5, then the function should return 1 + 2 + 3 + 4 + 5.\n\npublic class SumOfNumbers {", lang="java")
        response = stub.SendPrompt(request)
    print("response:", response)


if __name__ == "__main__":
    # main()
    code = "\n    public static int sumOf(int n) {\n        int sum = 0;\n\n        for (int i = 1; i <= n; i++) {\n            sum += i;\n        }\n\n        return sum;\n    }\n\n    public static void main(String[] args) {\n        assert sumOf(5) =="
    print(code.split("\n"))
