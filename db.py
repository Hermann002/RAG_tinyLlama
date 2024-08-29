from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")

def vector_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    vectorStore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./vector_store_db")

    return vectorStore

while True:
    
    vectorStore = Chroma(persist_directory="./vector_store_db", embedding_function=embeddings)

    pdf_path = input("Enter pdf path ")

    vectorStore.add_documents(vector_docs(pdf_path))


# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
# from langchain_chroma import Chroma
# from langchain_community.embeddings import OllamaEmbeddings

# embeddings = OllamaEmbeddings(model="llama3")

# webpage_url = input("Enter webpage URL")

# loader = WebBaseLoader(webpage_url)
# docs = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
# splits = text_splitter.split_documents(docs)
# vectorStore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./vector_store_db")