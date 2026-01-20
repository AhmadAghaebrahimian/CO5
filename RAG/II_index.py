# -*- coding: utf-8 -*-
"""
Task Sparse IR

Created on Tue November 11 10:28:17 2025

@author: agha
"""

import os
import glob
import chromadb
from typing import List
from I_constants import *
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, TextLoader



# Initialize folders
os.makedirs(models_path, exist_ok=True)

# Initialize vector database
chroma_client = chromadb.PersistentClient(persist_path)

LOADER_MAPPING = {
    "doc": (UnstructuredWordDocumentLoader, {}),
    "docx": (UnstructuredWordDocumentLoader, {}),
    "txt": (TextLoader, {"encoding": "utf8"}),
}

def load_single_document(file_path: str) -> (List[Document]):
    ext = file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        content = loader.load()
        return content
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_all_documents(source_path: str) -> List[Document]:
    all_files = []
    output = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_path, f"**/*{ext}"), recursive=True)
        )
    for file in all_files:
        output.extend(load_single_document(file))
    return output

def split_documents(source_path) -> List[Document]:
    documents = load_all_documents(source_path)
    if not documents:
        print("No new documents to load")
        texts = None
    else:
        print(f"Loaded {len(documents)} documents from {source_path}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each) from {len(documents)} documents.")
    return texts

def index():
    print(f"Indexing in progress ...!")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, cache_folder=models_path)
    split_texts = split_documents(source_path=source_path)
    print(len(split_texts))
    Chroma.from_documents(
        documents=split_texts,
        embedding=embeddings,
        collection_name="test_collection",
        persist_directory=persist_path,
        client=chroma_client
    )
    print(f"Indexing complete!")


if __name__ == "__main__":
    index()
