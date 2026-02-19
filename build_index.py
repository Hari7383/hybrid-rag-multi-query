import os
import pickle
from rank_bm25 import BM25Okapi

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ==============================
# CONFIG
# ==============================

DATASET_PATH = "datasetFED/"
PERSIST_DIR = "./FEDcoma_db"
BM25_PATH = "./bm25.pkl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


print("Loading documents...")

loader = DirectoryLoader(
    DATASET_PATH,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

split_docs = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(split_docs)}")


# ==============================
# CREATE EMBEDDINGS
# ==============================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory=PERSIST_DIR
)

vectorstore.persist()

print("Chroma DB created successfully.")

# if not os.path.exists(PERSIST_DIR):
#     vectorstore = Chroma.from_documents(
#         documents=split_docs,
#         embedding=embedding_model,
#         persist_directory=PERSIST_DIR
#     )
#     vectorstore.persist()
#     print("✅ Documents embedded and stored.")
# else:
#     vectorstore = Chroma(
#         persist_directory=PERSIST_DIR,
#         embedding_function=embedding_model
#     )
#     print("✅ Loaded existing Chroma DB.")


# ==============================
# BUILD BM25 INDEX
# ==============================

corpus = [doc.page_content for doc in split_docs]
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

with open(BM25_PATH, "wb") as f:
    pickle.dump({
        "bm25": bm25,
        "documents": split_docs
    }, f)

print("BM25 index saved successfully.")
print("Indexing complete.")