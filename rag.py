import pickle
import requests


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder


# ==============================
# CONFIG
# ==============================

PERSIST_DIR = "./FEDcoma_db"
BM25_PATH = "./bm25.pkl"

CHAT_URL = "API"
LLM_MODEL = "QuantTrio/Qwen3-VL-32B-Instruct-AWQ"

TOP_K_DENSE = 6
TOP_K_SPARSE = 6
FINAL_TOP_K = 5


print("Loading vector database...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_DENSE})


print("Loading BM25 index...")

with open(BM25_PATH, "rb") as f:
    data = pickle.load(f)

bm25 = data["bm25"]
split_docs = data["documents"]

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ==============================
# MULTI QUERY EXPANSION
# ==============================

def expand_query_multi(query):

    prompt = f"""
Generate 3 alternative search queries for:

"{query}"

Return each on a new line.
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Improve search queries."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    r = requests.post(CHAT_URL, json=payload, timeout=60)
    r.raise_for_status()

    content = r.json()["choices"][0]["message"]["content"]

    queries = [
        line.strip("- ").strip()
        for line in content.split("\n")
        if line.strip()
    ]

    return list(set([query] + queries))


# ==============================
# HYBRID RETRIEVAL
# ==============================

def hybrid_retrieve_multi(query):

    expanded_queries = expand_query_multi(query)

    all_docs = []

    for q in expanded_queries:

        dense_docs = retriever.invoke(q)

        tokenized_query = q.split()
        sparse_scores = bm25.get_scores(tokenized_query)

        top_sparse_idx = sorted(
            range(len(sparse_scores)),
            key=lambda i: sparse_scores[i],
            reverse=True
        )[:TOP_K_SPARSE]

        sparse_docs = [split_docs[i] for i in top_sparse_idx]

        all_docs.extend(dense_docs)
        all_docs.extend(sparse_docs)

    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    pairs = [(query, doc.page_content) for doc in unique_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, unique_docs),
        reverse=True,
        key=lambda x: x[0]
    )

    return [doc for _, doc in ranked[:FINAL_TOP_K]]


# ==============================
# LLM ANSWER
# ==============================

def ask_llm(docs, query):

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer strictly using the provided context.
If not found, reply exactly:
"Not found in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Answer strictly from document context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    r = requests.post(CHAT_URL, json=payload, timeout=120)
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"].strip()


# ==============================
# INTERACTIVE LOOP
# ==============================

print("\nHybrid RAG ready. Type 'exit' to quit.")

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    docs = hybrid_retrieve_multi(query)

    answer = ask_llm(docs, query)

    print("\nAnswer:\n")
    print(answer)
    print()