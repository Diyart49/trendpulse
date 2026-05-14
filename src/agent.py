import json
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from config import LLM_MODEL, GROQ_API_KEY
from src.fetcher import fetch_articles
from src.embedder import embed_and_store, search_similar
from src.analyzer import cluster_articles, analyze_with_llm

_agent_executor = None

SYSTEM_PROMPT = """You are TrendPulse, an intelligent news analysis agent.
You help users understand trending topics, track narratives, detect media bias, and identify contradictions in news coverage.
Lead with the dominant narrative and overall sentiment. Highlight contradictions if they exist. Be concise but insightful."""

@tool
def fetch_and_analyze_topic(topic: str) -> str:
    """Fetch latest news for a topic, cluster by narrative, and return full analysis with sentiment and contradictions."""
    articles = fetch_articles(topic, days_back=7)
    if not articles:
        return json.dumps({"error": f"No articles found for '{topic}'."})
    embeddings = embed_and_store(articles)
    articles = cluster_articles(articles, embeddings)
    analysis = analyze_with_llm(articles, topic)
    return json.dumps({
        "topic": topic, "total_articles": analysis["total_articles"],
        "dominant_narrative": analysis["meta"]["dominant_narrative"],
        "overall_sentiment": analysis["meta"]["overall_sentiment"],
        "contradictions": analysis["meta"]["contradictions"],
        "clusters": {str(k): {"narrative": v["narrative"], "sentiment": v["sentiment"],
                               "sentiment_score": v["sentiment_score"], "tone": v["tone"],
                               "article_count": v["size"], "key_claims": v["key_claims"]}
                     for k, v in analysis["clusters"].items()},
    }, indent=2)

@tool
def search_stored_articles(query: str) -> str:
    """Search ChromaDB for articles semantically similar to a query."""
    results = search_similar(query, n_results=6)
    if not results:
        return "No relevant articles found. Try fetching the topic first."
    return "\n".join([f"[{r['metadata'].get('source','?')}] {r['metadata'].get('title', r['text'][:80])} | {r['metadata'].get('published_at','')[:10]} | similarity: {r['similarity']:.2f}"
                      for r in results])

@tool
def compare_topics(topic1: str, topic2: str) -> str:
    """Compare stored articles from two different topics side by side."""
    def fmt(results, label):
        if not results:
            return f"{label}: No articles stored yet."
        return label + ":\n" + "\n".join([f"  - {r['text'][:150]}" for r in results])
    r1 = search_similar(topic1, n_results=4, topic_filter=topic1)
    r2 = search_similar(topic2, n_results=4, topic_filter=topic2)
    return f"{fmt(r1, topic1)}\n\n{fmt(r2, topic2)}"

def get_agent():
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)
    tools = [fetch_and_analyze_topic, search_stored_articles, compare_topics]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    _agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=4)
    return _agent_executor

def chat(user_message: str, history: list) -> str:
    agent = get_agent()
    lc_history = []
    for human, ai in history:
        lc_history.append(HumanMessage(content=human))
        lc_history.append(AIMessage(content=ai))
    response = agent.invoke({"input": user_message, "chat_history": lc_history})
    return response["output"]
