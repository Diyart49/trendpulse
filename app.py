import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.fetcher import fetch_articles
from src.embedder import embed_and_store
from src.analyzer import cluster_articles, analyze_with_llm
from src.agent import chat

def sentiment_color(score):
    if score > 0.1: return "#16A34A"
    if score < -0.1: return "#DC2626"
    return "#6B7280"

def build_summary(analysis, topic):
    meta = analysis["meta"]
    emoji = {"positive":"🟢","negative":"🔴","neutral":"⚪","divided":"🟡"}.get(meta["overall_sentiment"],"⚪")
    md = f"## {topic}\n**{analysis['total_articles']} articles analysed** · {emoji} **{meta['overall_sentiment'].title()} sentiment**\n\n> {meta['dominant_narrative']}\n"
    if meta.get("contradictions"):
        md += "\n**⚡ Contradictions detected:**\n" + "".join([f"- {c}\n" for c in meta["contradictions"]])
    md += "\n---\n"
    for k, v in analysis["clusters"].items():
        md += f"\n### Cluster {int(k)+1} — {v['tone'].title()}\n{v['narrative']}\n\n**Sentiment:** {v['sentiment']} · **Articles:** {v['size']}\n\n"
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
    fig_s = go.Figure(go.Bar(
        x=cluster_labels, y=scores,
        marker_color=[sentiment_color(s) for s in scores],
        text=[f"{s:+.2f}" for s in scores],
        textposition="outside",
        width=0.5
    ))
    fig_s.update_layout(
        title=dict(text="Sentiment Score per Narrative Cluster", font=dict(size=14, color="#111827")),
        yaxis=dict(range=[-1.3,1.3], gridcolor="#F3F4F6", title="Score"),
        xaxis=dict(gridcolor="#F3F4F6"),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        height=320, margin=dict(l=40,r=40,t=50,b=40),
        font=dict(family="Inter, sans-serif", color="#374151")
    )
    fig_s.add_hline(y=0, line_dash="dot", line_color="#D1D5DB", line_width=1)
    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    fig_t = px.line(daily, x="date", y="count", title="Article Volume Over Time",
                    color_discrete_sequence=["#2563EB"], markers=True)
    fig_t.update_traces(line=dict(width=2), marker=dict(size=6, color="#2563EB"))
    fig_t.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        height=280, margin=dict(l=40,r=40,t=50,b=40),
        yaxis=dict(title="Articles", gridcolor="#F3F4F6"),
        xaxis=dict(title="", gridcolor="#F3F4F6"),
        font=dict(family="Inter, sans-serif", color="#374151"),
        title=dict(font=dict(size=14, color="#111827"))
    )
    table_rows = [[a["title"][:90], a["source"], a["published_at"][:10], f"Cluster {a['cluster']+1}"] for a in articles]
    return summary_md, fig_s, fig_t, table_rows

CSS = """
body, .gradio-container { background: #F9FAFB !important; font-family: 'Inter', sans-serif !important; }
.gr-button-primary { background: #2563EB !important; border: none !important; border-radius: 8px !important; }
.gr-button-primary:hover { background: #1D4ED8 !important; }
.gr-textbox textarea, .gr-textbox input { border-radius: 8px !important; border: 1px solid #E5E7EB !important; background: #FFFFFF !important; }
.gr-slider { accent-color: #2563EB !important; }
.gr-tab { border-radius: 8px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="TrendPulse", css=CSS, theme=gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "sans-serif"]
)) as demo:

    gr.HTML("""
    <div style='padding: 28px 0 8px 0;'>
      <div style='display:flex; align-items:center; gap:12px; margin-bottom:6px;'>
        <span style='font-size:32px;'>📡</span>
        <span style='font-size:26px; font-weight:700; color:#111827; letter-spacing:-0.5px;'>TrendPulse</span>
      </div>
      <p style='color:#6B7280; font-size:14px; margin:0;'>
        News intelligence agent — track narratives, detect sentiment shifts, and surface contradictions across sources.
      </p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("🔍  Analyse Topic"):
            gr.HTML("<div style='height:8px'></div>")
            with gr.Row(equal_height=True):
                topic_input = gr.Textbox(
                    label="Topic",
                    placeholder="e.g.  OpenAI  ·  Indian economy  ·  climate policy  ·  Fed rate cuts",
                    scale=5, container=True
                )
                days_slider = gr.Slider(1, 30, value=7, step=1, label="Days back", scale=1)

            analyze_btn = gr.Button("Analyse  →", variant="primary", size="lg")

            gr.HTML("<div style='height:4px'></div>")
            summary_output = gr.Markdown()

            with gr.Row():
                sentiment_plot = gr.Plot(show_label=False)
                timeline_plot = gr.Plot(show_label=False)

            articles_table = gr.Dataframe(
                headers=["Title", "Source", "Date", "Cluster"],
                label="Articles", wrap=True, row_count=10
            )

            analyze_btn.click(
                fn=analyze_topic,
                inputs=[topic_input, days_slider],
                outputs=[summary_output, sentiment_plot, timeline_plot, articles_table]
            )

        with gr.Tab("💬  Chat with Agent"):
            gr.HTML("""
            <p style='color:#6B7280; font-size:13px; margin: 12px 0 4px 0;'>
            Ask the agent to fetch, analyse, or compare any topic. It remembers your conversation.
            </p>""")
            gr.ChatInterface(
                fn=chat,
                
                examples=[
                    "What are the dominant narratives around AI regulation right now?",
                    "Fetch news on the Indian economy and summarise the key stories.",
                    "Compare sentiment between coverage of OpenAI and Google DeepMind.",
                    "What contradictions exist in how outlets are covering the US economy?",
                ],
                title=""
            )

    gr.HTML("""
    <div style='text-align:center; padding: 16px 0 8px; color:#9CA3AF; font-size:12px;'>
        Built with LangChain · HuggingFace · ChromaDB · Groq · Gradio
    </div>""")

if __name__ == "__main__":
    demo.launch()
