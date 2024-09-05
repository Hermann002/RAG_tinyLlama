import os
from decouple import config

from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mistralai import Mistral


api_key = config("MISTRAL_API_KEY")
os.environ['HF_TOKEN'] = config("HF_TOKEN")

# embeddings = OllamaEmbeddings(model="llama3")
# embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = Mistral(api_key=api_key)

def vector_docs(pdf_path):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    documents = text_splitter.split_documents(docs)
    
    try:
        vectorStore = FAISS.load_local("faiss.index", embeddings, allow_dangerous_deserialization=True)
        vectorStore.add_documents(documents=documents)
        vectorStore.save_local("faiss.index")
        print("Création d'une nouvelle base de données vectorielle")
    except Exception as e:
        try : 
            vectorStore = FAISS.from_documents(documents, embeddings)
        except Exception as e:
            vectorStore = FAISS.from_documents(documents, embeddings)
        vectorStore.save_local("faiss.index")
        print("Ajout des données à la base de données vectorielle")
    return vectorStore

while True:
    pdf_path = input("Enter pdf path ")
    
    vector_docs(pdf_path)