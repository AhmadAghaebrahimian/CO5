# -*- coding: utf-8 -*-
"""
Task Sparse IR

Created on Tue November 11 09:14:24 2025

@author: agha
"""

# Indexer
persist_path= 'chroma_db'
models_path = 'models'
source_path = '../squad/texts'
embeddings_model_name='intfloat/multilingual-e5-base'
chunk_size = 500
chunk_overlap = 50

# LLM generator
llm_base = 'llms'
model_n_ctx=512
model_n_batch=32
K_source_chunks=3
num_gpu_layers =20
