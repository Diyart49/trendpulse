from newsapi import NewsApiClient
from datetime import datetime, timedelta
from config import NEWS_API_KEY, MAX_ARTICLES
import hashlib

def fetch_articles(topic: str, days_back: int = 7) -> list[dict]:
    client = NewsApiClient(api_key=NEWS_API_KEY)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    response = client.get_everything(
        q=topic, from_param=from_date, language="en",
        sort_by="relevancy", page_size=MAX_ARTICLES,
    )
    articles = []
    for article in response.get("articles", []):
        title = article.get("title", "")
        description = article.get("description", "")
        if not title or not description:
            continue
        article_id = hashlib.md5(article["url"].encode()).hexdigest()[:12]
        articles.append({
            "id": article_id, "title": title, "description": description,
            "source": article["source"]["name"], "url": article["url"],
            "published_at": article["publishedAt"],
            "text": f"{title}. {description}",
            "topic": topic.lower().strip(),
        })
    return articles
