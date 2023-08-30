#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/12 15:08
# @Author  : Jack
# @File    : ingest.py
# @Software: PyCharm

""" Load files, clean up, split, ingest into VectorStore"""
import os
from pathlib import Path

from langchain import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# Fill your openai api key.
os.environ["OPENAI_API_KEY"] = ""


def ingest_mds(source_dir: str, store_dir: str):
    """Find all markdown files from source_dir, ingest into VectorStore """

    # Find all markdown files
    markdown_files = Path(source_dir).glob("**/*.md")
    data = []
    sources = []
    for m in markdown_files:
        with open(m, encoding='utf8') as f:
            data.append(f.read())
        sources.append(m)

    # Here we split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": sources[i]}] * len(splits))

    # Here we create a vector store from the documents and save it to disk.
    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    store.save_local(store_dir)


if __name__ == '__main__':
    # Fill in your document address and the directory address where the vector database is saved.
    ingest_mds(r"docs-dir", "vector_store")
