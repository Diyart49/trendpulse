import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.fetcher import fetch_articles
from src.embedder import embed_and_store
from src.analyzer import cluster_articles, analyze_with_llm
from src.agent import chat

def sentiment_color(score):
    if score > 0.1: return "#1D9E75"
    if score < -0.1: return "#D85A30"
    return "#888780"

def build_summary(analysis, topic):
    meta = analysis["meta"]
    emoji = {"positive":"🟢","negative":"🔴","neutral":"⚪","divided":"🟡"}.get(meta["overall_sentiment"],"⚪")
    md = f"## {topic}\n**{analysis['total_articles']} articles** · {emoji} **{meta['overall_sentiment'].title()} sentiment**\n\n> {meta['dominant_narrative']}\n"
    if meta.get("contradictions"):
        md += "\n**⚡ Contradictions detected:**\n" + "".join([f"- {c}\n" for c in meta["contradictions"]])
    md += "\n---\n"
    for k, v in analysis["clusters"].items():
        md += f"\n### Cluster {int(k)+1} — *{v['tone'].title()}*\n{v['narrative']}\n\n**Sentiment:** {v['sentiment']} · **Articles:** {v['size']}\n\n"
        if v.get("key_claims"):
            md += "**Key claims:**\n" + "".join([f"- {c}\n" for c in v["key_claims"]])
    return md

def analyze_topic(topic, days_back):
    if not topic.strip():
        return "Please enter a topic.", None, None, []
    articles = fetch_articles(topic.strip(), days_back=int(days_back))
    if not articles:
        return f"⚠️ No articles found for **'{topic}'**. Try a broader term.", None, None, []
    embeddings = embed_and_store(articles)
    articles = cluster_articles(articles, embeddings)
    analysis = analyze_with_llm(articles, topic)
    summary_md = build_summary(analysis, topic)
    cluster_labels = [f"Cluster {int(k)+1}" for k in analysis["clusters"].keys()]
    scores = [v["sentiment_score"] for v in analysis["clusters"].values()]
    fig_s = go.Figure(go.Bar(x=cluster_labels, y=scores, marker_color=[sentiment_color(s) for s in scores],
                             text=[f"{s:+.2f}" for s in scores], textposition="outside"))
    fig_s.update_layout(title="Sentiment per Cluster", yaxis=dict(range=[-1.3,1.3]),
                        plot_bgcolor="white", paper_bgcolor="white", height=300, margin=dict(l=40,r=40,t=50,b=40))
    fig_s.add_hline(y=0, line_dash="dash", line_color="#cccccc")
    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    fig_t = px.area(daily, x="date", y="count", title="Article Volume Over Time", color_discrete_sequence=["#378ADD"])
    fig_t.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=260, margin=dict(l=40,r=40,t=50,b=40))
    table_rows = [[a["title"][:90], a["source"], a["published_at"][:10], f"Cluster {a['cluster']+1}"] for a in articles]
    return summary_md, fig_s, fig_t, table_rows

with gr.Blocks(title="TrendPulse", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 📡 TrendPulse\n*News Intelligence & Narrative Tracking Agent*")
    with gr.Tabs():
        with gr.Tab("🔍 Analyse Topic"):
            with gr.Row():
                topic_input = gr.Textbox(label="Topic", placeholder="e.g. OpenAI, Indian economy, climate policy...", scale=4)
                days_slider = gr.Slider(1, 30, value=7, step=1, label="Days back", scale=1)
            analyze_btn = gr.Button("Analyse →", variant="primary", size="lg")
            summary_output = gr.Markdown()
            with gr.Row():
                sentiment_plot = gr.Plot()
                timeline_plot = gr.Plot()
            articles_table = gr.Dataframe(headers=["Title","Source","Date","Cluster"], wrap=True)
            analyze_btn.click(fn=analyze_topic, inputs=[topic_input, days_slider],
                              outputs=[summary_output, sentiment_plot, timeline_plot, articles_table])
        with gr.Tab("💬 Chat with Agent"):
            gr.Markdown("Ask the agent to fetch, analyse, or compare any topic.")
            gr.ChatInterface(fn=chat, examples=[
                "What are the dominant narratives around AI regulation?",
                "Fetch news on the Indian economy and summarise.",
                "Compare sentiment between OpenAI and Google DeepMind.",
            ], title="")

if __name__ == "__main__":
    demo.launch()
