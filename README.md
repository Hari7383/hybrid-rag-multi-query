#  Hybrid RAG with Multi-Query Expansion

A production-style Retrieval-Augmented Generation (RAG) system combining:

- 🔹 Dense Retrieval (Sentence Transformers)
- 🔹 Sparse Retrieval (BM25)
- 🔹 Cross-Encoder Reranking
- 🔹 Multi-Query Expansion
- 🔹 Strict Grounded LLM Answering
- 🔹 Persistent Chroma Vector Database

This project demonstrates an enterprise-level document QA pipeline designed to handle vocabulary mismatch, paraphrased queries, and structured documents.

---

##  Architecture

User Query  
→ Multi-Query Expansion (LLM)  
→ Hybrid Retrieval (Dense + BM25)  
→ Cross-Encoder Reranking  
→ Grounded LLM Answer  

---

##  Key Features

-  Multi-query expansion for better semantic matching
-  Hybrid dense + sparse retrieval for improved recall
-  Cross-encoder reranking for precision
-  Persistent Chroma vector database
-  Strict document-grounded responses (no hallucination)
-  Modular design (separate indexing and querying scripts)

---

##  Project Structure

hybrid-rag-multi-query/

│

├── build_index.py   # Offline indexing script

├── query_rag.py     # Query and answer script

├── datasetFED/      # PDF dataset

├── FEDcoma_db/      # Chroma vector database (generated)

├── bm25.pkl         # BM25 index (generated)

├── requirements.txt

└── README.md

---

##  Installation

```bash
pip install -r requirements.txt
```
Or manually:
```
pip install langchain langchain-community langchain-core
pip install langchain-huggingface
pip install sentence-transformers
pip install chromadb
pip install rank-bm25
pip install pypdf
pip install requests
```

---

Step 1: Build Vector Index
```
python build_index.py
```

This will:

- Load PDFs

- Chunk documents

- Create embeddings

- Store in Chroma

- Build BM25 index

Step 2: Query the System
```
python query_rag.py
```

Ask natural language questions directly from your terminal.

---

## Example Queries
```
What are the key sections of this document?
Explain the filing requirements.
How is taxable income calculated?
What does Line 16 refer to?
```

---

## Why Hybrid Retrieval?

- Dense retrieval handles semantic similarity.
- BM25 handles exact keyword and numeric matching.
- Cross-encoder reranking improves final precision.
- Combining all three drastically improves retrieval performance over traditional RAG pipelines.

---

## Improvements Over Basic RAG

| Basic RAG                | This System             |
| ------------------------ | ----------------------- |
| Single query retrieval   | Multi-query expansion   |
| Dense-only search        | Dense + Sparse hybrid   |
| No reranking             | Cross-encoder reranking |
| High vocabulary mismatch | Improved recall         |
| Hallucination risk       | Strict grounded prompts |

---

## Tech Stack

- LangChain

- ChromaDB

- Sentence Transformers

- BM25 (rank-bm25)

- Cross-Encoder (MS MARCO)

- Custom LLM endpoint

---

## Future Improvements

- Retrieval confidence scoring

- Reciprocal Rank Fusion (RRF)

- FastAPI deployment

- Caching layer

- Evaluation framework

- Dockerization

---

## License

MIT License
