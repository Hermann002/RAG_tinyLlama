
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")

while True:
    try:
        vectorStore = Chroma(persist_directory="./vector_store_db", embedding_function=embeddings)

        pdf_path = input("Enter pdf path ")

        loader = WebBaseLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        splits = text_splitter.split_documents(docs)
        vectorStore_ = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./vector_store_db")
        vectorStore.add_documents(vectorStore_)
    except Exception as e:
        print(e)
    pdf_path = input("Enter pdf path ")

    loader = WebBaseLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    vectorStore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./vector_store_db")
