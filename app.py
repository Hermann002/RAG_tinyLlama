import time
import os
from decouple import config

from mistralai import Mistral

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from langchain_mistralai.embeddings import MistralAIEmbeddings

st.title("Chat with Documents")
st.caption("this app allows you to chat with document create by expert in data engineering")

api_key = config('MISTRAL_API_KEY')



client = Mistral(api_key=api_key)

embedding_mistral = client.embeddings.create(
    model="mistral-embed",
    inputs=["Embed this sentence.", "As well as this one."],
)

# embeddings = OllamaEmbeddings(model="llama3")
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
vectorStore = Chroma(persist_directory="./vector_store_db", embedding_function=embedding_mistral)
retriever = vectorStore.as_retriever()



def ollama_llm(question, context):
    formatted_prompt = f"question: {question}\n\n context: {context}"
    # response = ollama.chat(model="llama3", messages=[{'role': 'user', "content": formatted_prompt}])
    response = client.chat.complete(model= 'open-mistral-7b',
    messages = [
        {
            "role": "user",
            "content": formatted_prompt,
        },
    ])
    return response['message']['content']


def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)


prompt = st.text_input("ask any question about the webpage") + "Répond moi en Français"

if prompt:
    start_time = time.time()
    result = rag_chain(prompt)
    st.write(retriever.invoke(prompt))
    st.write(result)
    end_time = time.time()

    execution_time = end_time - start_time

    st.write(f"Le temps de réponse est de {execution_time}s")
