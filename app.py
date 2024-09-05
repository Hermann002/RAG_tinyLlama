import time
import os
from decouple import config

from mistralai import Mistral

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from mistralai import Mistral

st.title("Chat with Documents")
st.caption("this app allows you to chat with document create by expert in data engineering")

try:
    os.environ.pop('MISTRAL_API_KEY')
except Exception:
     pass
api_key = config('MISTRAL_API_KEY')

client = Mistral(api_key=api_key)
# embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorStore = FAISS.load_local("faiss.index", embeddings, allow_dangerous_deserialization=True)

retriever = vectorStore.as_retriever()
model = "mistral-large-latest"


def _llm(question, context):
    formatted_prompt = f"question: {question}\n\n context: {context}"
    response = client.chat.complete(
        model=model,
        messages=[{
            "role": "user",
            "content" : formatted_prompt
        }])
    return response.choices[0].message.content


def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return _llm(question, formatted_context)


prompt = st.text_input("ask any question about the webpage")

if prompt:
    start_time = time.time()
    result = rag_chain(prompt)
    st.write(retriever.invoke(prompt))
    st.write(result)
    end_time = time.time()

    execution_time = end_time - start_time

    st.write(f"Le temps de réponse est de {execution_time}s")
