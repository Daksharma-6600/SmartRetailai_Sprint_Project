from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

import os

# =========================
# LOAD PDF DOCUMENTS
# =========================

documents = []

pdf_folder = "documents"

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        path = os.path.join(pdf_folder, file)

        loader = PyPDFLoader(path)

        docs = loader.load()

        documents.extend(docs)

print("PDFs Loaded Successfully")

# =========================
# TEXT SPLITTING
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

print("Text Chunks Created")

# =========================
# EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# VECTOR DATABASE
# =========================

vectorstore = FAISS.from_documents(
    texts,
    embeddings
)

print("FAISS Vector Store Created")

# =========================
# SAVE VECTORSTORE
# =========================

vectorstore.save_local("faiss_index")

print("Vector Store Saved")