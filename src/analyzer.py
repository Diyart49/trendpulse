import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import LLM_MODEL, GROQ_API_KEY, N_CLUSTERS

def cluster_articles(articles: list[dict], embeddings: np.ndarray) -> list[dict]:
    n = min(N_CLUSTERS, len(articles))
    if n < 2:
        for a in articles:
            a["cluster"] = 0
        return articles
    normalized = normalize(embeddings)
    labels = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(normalized)
    for i, article in enumerate(articles):
        article["cluster"] = int(labels[i])
    return articles

def _parse_json(text: str, fallback: dict) -> dict:
    try:
        return json.loads(text.strip())
    except:
        return fallback

def analyze_with_llm(articles: list[dict], topic: str) -> dict:
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)
    clusters: dict[int, list[dict]] = {}
    for article in articles:
        clusters.setdefault(article.get("cluster", 0), []).append(article)

    cluster_analyses = {}
    for cluster_id, cluster_articles in clusters.items():
        text_block = "\n".join([f"- [{a['source']}] {a['text']}" for a in cluster_articles[:6]])
        prompt = f"""Analyse these news articles about "{topic}":
{text_block}
Respond ONLY with valid JSON (no markdown):
{{"narrative": "<1-sentence dominant narrative>", "sentiment": "<positive|negative|neutral|mixed>",
"sentiment_score": <float -1.0 to 1.0>, "key_claims": ["<claim1>", "<claim2>", "<claim3>"], "tone": "<1-2 words>"}}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        analysis = _parse_json(response.content, {"narrative": "Mixed coverage", "sentiment": "neutral",
                                                    "sentiment_score": 0.0, "key_claims": [], "tone": "neutral"})
        cluster_analyses[cluster_id] = {**analysis, "articles": cluster_articles, "size": len(cluster_articles)}

    narratives = "\n".join([f"Cluster {k}: {v['narrative']}" for k, v in cluster_analyses.items()])
    meta_prompt = f"""News coverage clusters for "{topic}":
{narratives}
Respond ONLY with valid JSON (no markdown):
{{"contradictions": ["<c1>", "<c2>"], "overall_sentiment": "<positive|negative|neutral|divided>",
"dominant_narrative": "<1-sentence summary>"}}"""
    meta_response = llm.invoke([HumanMessage(content=meta_prompt)])
    meta = _parse_json(meta_response.content, {"contradictions": [], "overall_sentiment": "neutral",
                                                "dominant_narrative": "Diverse coverage detected."})
    return {"clusters": cluster_analyses, "meta": meta, "topic": topic, "total_articles": len(articles)}
