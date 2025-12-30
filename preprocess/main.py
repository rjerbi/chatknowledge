import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



pdf_folder = "blobs"
documents = []

print("Chargement des fichiers PDF...")
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        try:
            print(f"Lecture de : {file}")
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            docs = loader.load()
            print(f"{len(docs)} pages chargées depuis {file}.")
            documents.extend(docs)
        except Exception as e:
            print(f"Erreur lors du chargement de {file} : {e}")


print("Découpage du contenu en morceaux...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"Nombre total de morceaux générés : {len(chunks)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chargement du modèle d'embedding sur : {device}")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)


print("Création de la base vectorielle FAISS...")
vectorstore = FAISS.from_documents(chunks, embeddings)


vectorstore.save_local("vector_db")
print("Base vectorielle créée et sauvegardée.")