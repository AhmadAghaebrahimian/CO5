# -*- coding: utf-8 -*-
"""
Task Sparse IR

Created on Tue November 11 04:52:43 2025

@author: agha
"""

import chromadb
import gradio as gr
from I_constants import *
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# 1. Instantiate the tokenizer and model with a model id
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Uses GPU if available, if only cpu: device_map={"": "cpu"}
    torch_dtype="auto"  # Automatically sets precision
)

# 2. Create a Transformers pipeline and wrap it up with HuggingFacePipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)
hf_llm = HuggingFacePipeline(pipeline=generator)

# 3. Load the index and instantiate the retriever
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, cache_folder=models_path)
chroma_client = chromadb.PersistentClient(persist_path)
db = Chroma(persist_directory=persist_path,
            embedding_function=embeddings,
            collection_name="test_collection",
            client=chroma_client
            )

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": K_source_chunks})


# 4. Basic prompt for RAG
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# 5. Connect (chain) the retriever and LLM generator
qa_llm = RetrievalQA.from_chain_type(
    llm=hf_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

def main():
    def sop_qa(question):
        return qa_llm.invoke(question)['result'].split("Answer:")[1]

    # 6. (Optional) UI
    with gr.Blocks() as demo:
        demo.css = "footer {visibility: hidden}"
        demo.title = 'CO5: Basic RAG'
        html_box_title = gr.HTML('<h1>CO5: Basic RAG</h1>')
        html_box = gr.HTML('<h3>This is a basic RAG system for QA application</h3>')

        with gr.Row(equal_height=False):
            with gr.Column() as col_right:
                text_box = gr.Textbox(label='Ask:', lines=2, placeholder="Ask")
                with gr.Row():
                    source_btn = gr.Button("Sources", variant="primary")
                    answer_btn = gr.Button("Answer", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")
                error_box = gr.Textbox(label="Error", visible=False)
                info_box = gr.Textbox(label="Info", visible=False)

            with gr.Column() as left:
                with gr.Column() as answer_col:
                    answer_box = gr.Textbox(label='Answer', lines=2)
                    accept_btn = gr.Button("Accept", size='sm', variant="primary")
                    reject_btn = gr.Button("Reject", size='sm')
                with gr.Column() as source_col:
                    source_1 = gr.Textbox(label='Source 1', lines=2)
                    source_2 = gr.Textbox(label='Source 2', lines=2)
                    source_3 = gr.Textbox(label='Source 3', lines=2)


        def submit_answer(question):
            if question is None or len(question) < 3:
                return {error_box: gr.update(value="No proper question!", visible=True)}
            ans = sop_qa(question)
            return {
                answer_col: gr.update(visible=True),
                error_box : gr.update(visible=False),
                info_box: gr.update(visible=False),
                answer_box: ans
            }

        def submit_sources(question):
            if question is None or len(question) < 3:
                return {
                    info_box: gr.update(visible=False),
                    error_box: gr.update(value="No proper question!", visible=True)
                }

            src_1, src_2, src_3 = ["(%s:Page:/%d)\n%s" % (doc.metadata["source"], doc.metadata.get("page", 1), doc.page_content) for n, doc in enumerate(retriever.invoke(question)) if n<3]
            return {
                source_col: gr.update(visible=True),
                error_box: gr.update(visible=False),
                info_box: gr.update(visible=False),
                source_1: src_1,
                source_2: src_2,
                source_3: src_3,
            }

        def reset():
            return {
                info_box: gr.update(visible=False),
                error_box: gr.update(visible=False),
                source_1: None,
                source_2: None,
                source_3: None,
                text_box: None,
                answer_box: None,
            }


        def accept_answer(question, answer, model):
            if question is None or len(question) < 3:
                return {
                    info_box: gr.update(visible=False),
                    error_box: gr.update(value="No answer!", visible=True)
                }
            else:
                return {info_box: gr.update(value="The response recorded!", visible=True)}

        def reject_answer(question, answer, model):
            if question is None or len(question) < 3:
                return {error_box: gr.update(value="No answer!", visible=True)}
            else:
                return {info_box: gr.update(value="The response recorded!", visible=True)}

        source_btn.click(fn=submit_sources, inputs=[text_box],
                         outputs=[error_box, info_box, source_col, source_1, source_2, source_3])
        answer_btn.click(fn=submit_answer, inputs=[text_box],
                         outputs=[error_box, info_box, answer_col, answer_box])
        reset_btn.click(fn=reset, inputs=None, outputs=[text_box, error_box, info_box, answer_box, source_1, source_2, source_3])
        accept_btn.click(fn=accept_answer, inputs=[text_box, answer_box], outputs=[error_box, info_box])
        reject_btn.click(fn=reject_answer, inputs=[text_box, answer_box], outputs=[error_box, info_box])
    demo.launch(favicon_path='favicon.ico', show_api=False, width='20%', server_name="0.0.0.0", server_port=7000)


if __name__ == "__main__":
    main()
