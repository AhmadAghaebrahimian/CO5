# -*- coding: utf-8 -*-
"""
Task Sparse IR

Created on Tue November 11 10:42:53 2025

@author: agha
"""

import json
import chromadb
import numpy as np
from tqdm import tqdm
from I_constants import *
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

test_data = json.load(open('../squad/squad_multiple_contexts.json', 'r'))

def get_f1(true_doc: set, pred_doc: set) -> float:
    # compute F1 score

    return

if __name__ == "__main__":
    version = 1
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, cache_folder=models_path)
    chroma_client = chromadb.PersistentClient(persist_path)
    db = Chroma(persist_directory=persist_path,
                embedding_function=embeddings,
                collection_name="test_collection",
                client=chroma_client
                )

    retriever = db.as_retriever(search_type="similarity",
                                search_kwargs={"k": K_source_chunks})

    f1s = []
    for entry in tqdm(test_data):
        query = entry['text']
        entry_prediction = [doc.metadata["source"].split('/')[-1] for n, doc in enumerate(retriever.invoke(query))]
        entry_true = entry['sources']
        f1s.append(get_f1(set(entry_true), set(entry_prediction)))

    print("F1: %.2f"%(np.mean(f1s)))
