import ollama
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

st.title("Chat with Documents")
st.caption("this app allows you to chat with document create by expert in data engineering")

embeddings = OllamaEmbeddings(model="llama3")
vectorStore = Chroma(persist_directory="./vector_store_db", embedding_function=embeddings)
retriever = vectorStore.as_retriever()

def ollama_llm(question, context):
    formatted_prompt = f"question: {question}\n\n context: {context}"
    response = ollama.chat(model="llama3", messages=[{'role': 'user', "content": formatted_prompt}])
    return response['message']['content']


def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)


prompt = st.text_input("ask any question about the webpage")

if prompt:
    result = rag_chain(prompt)
    st.write(retriever)
    st.write(result)
