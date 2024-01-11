from langchain.llms import CTransformers
from langchain.text_splitter import CharacterTextSplitter

import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from ui import css, bot_template, user_template
from langchain.llms import CTransformers
from langchain import PromptTemplate
import langchain
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader

from dotenv import load_dotenv
load_dotenv()

llm = CTransformers(
                model="TheBloke/Llama-2-7B-Chat-GGML",
                model_type="llama",
                max_new_tokens = 512,
                temperature = 0.9
            )

hfembeddings = HuggingFaceEmbeddings(
                            model_name="thenlper/gte-large",
                            model_kwargs={'device': 'cpu'}
                        )

text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )

def create_vector_db(docs):
    print("docs",docs)
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print("text",text)
    print("length of text", len(text))

    chunks=text_splitter.split_text(text)
    print("length of chunks", len(chunks))

    vectorstore = FAISS.from_texts(texts=chunks, embedding=hfembeddings)
    vectorstore.save_local("faiss")
    print("done")

def get_qa_chain():
    vectorstore = FAISS.load_local("faiss", hfembeddings)

    prompt_temp = '''
With the information provided try to answer the question.
If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
This is related to medical domain. So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

Context: {context}
Question: {question}
Do provide only correct answers

Correct answer:
    '''

    custom_prompt_temp = PromptTemplate(template=prompt_temp,
                                input_variables=['context', 'question'])

    chain = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    chain_type="stuff",
                                    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": custom_prompt_temp})


    return chain 

# chain = get_qa_chain()
# print(chain("How to cure cough ?"))
# print(chain("how to cure heartburn ?"))