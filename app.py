import ollama
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from chromadb import Client
from chromadb.utils.embedding_functions import ollama_embedding_function

st.title("Chat with Documents")
st.caption("this app allows you to chat with document create by expert in data engineering")

if 'history' not in st.session_state:
    st.session_state.history = []
    
embeddings = OllamaEmbeddings(model="llama3")


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

webpage_url = st.text_input("Enter webpage URL", type="default")

if webpage_url:
    loader = WebBaseLoader(webpage_url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    
    client = Client()
    # collection = client.get_or_create_collection(name="my_collection")
    vectorStore = Chroma.from_documents(documents=splits, embedding=embeddings)
    # vectorStore = Chroma(client=client, embedding_function=embeddings, collection_name="my_collection")
    # vectorStore.add_texts(splits)

    
    
    # Rag Setup
    retriever = vectorStore.as_retriever()

    
    st.success(f"load {webpage_url} succesfully !")

prompt = st.text_input("ask any question about the webpage")

if prompt:
    result = rag_chain(prompt)
    st.write(result)
