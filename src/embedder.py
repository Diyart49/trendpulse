import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_collection(collection_name: str = "trendpulse"):
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

def embed_and_store(articles: list[dict]) -> np.ndarray:
    model = get_model()
    collection = get_collection()
    texts = [a["text"] for a in articles]
    embeddings = model.encode(texts, show_progress_bar=False)
    collection.upsert(
        ids=[a["id"] for a in articles],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"title": a["title"], "source": a["source"], "url": a["url"],
                    "published_at": a["published_at"], "topic": a["topic"]} for a in articles],
    )
    return embeddings

def search_similar(query: str, n_results: int = 5, topic_filter: str = None) -> list[dict]:
    model = get_model()
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []
    query_embedding = model.encode([query])[0].tolist()
    where = {"topic": topic_filter.lower().strip()} if topic_filter else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [{"id": results["ids"][0][i], "text": results["documents"][0][i],
             "metadata": results["metadatas"][0][i], "similarity": 1 - results["distances"][0][i]}
            for i in range(len(results["ids"][0]))]
