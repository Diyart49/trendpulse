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
    sentiment_label = meta["overall_sentiment"].title()
    md = f"## {topic}\n\n"
    md += f"**{analysis['total_articles']} articles analysed** &nbsp;·&nbsp; **{sentiment_label} sentiment**\n\n"
    md += f"> {meta['dominant_narrative']}\n\n"
    if meta.get("contradictions"):
        md += "**Contradictions detected:**\n"
        for c in meta["contradictions"]:
            md += f"- {c}\n"
        md += "\n"
    md += "---\n"
    for k, v in analysis["clusters"].items():
        md += f"\n### Cluster {int(k)+1} &mdash; {v['tone'].title()}\n"
        md += f"{v['narrative']}\n\n"
        md += f"**Sentiment:** {v['sentiment']} &nbsp;·&nbsp; **Articles:** {v['size']}\n\n"
        if v.get("key_claims"):
            md += "**Key claims:**\n"
            for c in v["key_claims"]:
                md += f"- {c}\n"
    return md


def analyze_topic(topic, days_back):
    if not topic.strip():
        return (
            "Please enter a topic.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    articles = fetch_articles(topic.strip(), days_back=int(days_back))
    if not articles:
        return (
            f"No articles found for '{topic}'. Try a broader term or extend the date range.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    embeddings = embed_and_store(articles)
    articles = cluster_articles(articles, embeddings)
    analysis = analyze_with_llm(articles, topic)
    summary_md = build_summary(analysis, topic)

    # Sentiment chart
    cluster_labels = [f"Cluster {int(k)+1}" for k in analysis["clusters"].keys()]
    scores = [v["sentiment_score"] for v in analysis["clusters"].values()]
    fig_s = go.Figure(go.Bar(
        x=cluster_labels, y=scores,
        marker_color=[sentiment_color(s) for s in scores],
        text=[f"{s:+.2f}" for s in scores],
        textposition="outside",
        width=0.45,
    ))
    fig_s.update_layout(
        title=dict(text="Sentiment per Cluster", font=dict(size=13, color="#111827")),
        yaxis=dict(range=[-1.4, 1.4], gridcolor="#E5E7EB", title="Score", zeroline=False),
        xaxis=dict(gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=300,
        margin=dict(l=40, r=40, t=44, b=40),
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
    )
    fig_s.add_hline(y=0, line_dash="dot", line_color="#D1D5DB", line_width=1)

    # Timeline chart
    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    fig_t = px.area(daily, x="date", y="count",
                    title="Article Volume Over Time",
                    color_discrete_sequence=["#2563EB"])
    fig_t.update_traces(line=dict(width=2, color="#2563EB"),
                        fillcolor="rgba(37,99,235,0.08)")
    fig_t.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=300,
        margin=dict(l=40, r=40, t=44, b=40),
        yaxis=dict(title="Articles", gridcolor="#E5E7EB", zeroline=False),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        title=dict(font=dict(size=13, color="#111827")),
    )

    table_rows = [
        [a["title"][:85], a["source"], a["published_at"][:10], f"Cluster {a['cluster']+1}"]
        for a in articles
    ]

    return (
        summary_md,
        gr.update(value=fig_s, visible=True),
        gr.update(value=fig_t, visible=True),
        gr.update(value=table_rows, visible=True),
        gr.update(visible=True),
    )


CSS = """
/* ── Base ── */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Input & button ── */
.gr-button-primary {
    background: #2563EB !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}
.gr-button-primary:hover { background: #1D4ED8 !important; }

/* ── Tab labels ── */
.tab-nav button {
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 8px 0 16px 0;
}

footer { display: none !important; }
"""

with gr.Blocks(
    title="TrendPulse",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=["Inter", "ui-sans-serif", "sans-serif"],
    ),
) as demo:

    gr.HTML("""
    <div style="padding: 24px 0 4px 0; border-bottom: 1px solid #E5E7EB; margin-bottom: 20px;">
        <div style="font-size: 22px; font-weight: 700; color: #111827; letter-spacing: -0.4px;">
            TrendPulse
        </div>
        <div style="font-size: 13px; color: #6B7280; margin-top: 4px;">
            Track narratives, detect sentiment shifts, and surface contradictions across news sources.
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Analyse ──────────────────────────────────────────
        with gr.Tab("Analyse Topic"):

            with gr.Row(equal_height=True):
                topic_input = gr.Textbox(
                    label="Topic",
                    placeholder="e.g. OpenAI  /  Indian economy  /  climate policy  /  Fed rate cuts",
                    scale=5,
                )
                days_slider = gr.Slider(
                    minimum=1, maximum=30, value=7, step=1,
                    label="Days back", scale=1,
                )

            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")

            # Results — hidden until analysis runs
            summary_output = gr.Markdown(visible=True)

            with gr.Row(visible=False) as charts_row:
                sentiment_plot = gr.Plot(show_label=False, visible=False)
                timeline_plot  = gr.Plot(show_label=False, visible=False)

            articles_table = gr.Dataframe(
                headers=["Title", "Source", "Date", "Cluster"],
                label="Articles",
                wrap=True,
                row_count=10,
                visible=False,
            )

            analyse_btn.click(
                fn=analyze_topic,
                inputs=[topic_input, days_slider],
                outputs=[
                    summary_output,
                    sentiment_plot,
                    timeline_plot,
                    articles_table,
                    charts_row,
                ],
            )

        # ── Tab 2: Chat ──────────────────────────────────────────────
        with gr.Tab("Chat with Agent"):
            gr.HTML("""
            <p style="font-size: 13px; color: #6B7280; margin: 12px 0 8px 0;">
                Ask the agent to fetch, analyse, or compare any topic. It remembers your conversation context.
            </p>
            """)
            gr.ChatInterface(
                fn=chat,
                examples=[
                    "What are the dominant narratives around AI regulation right now?",
                    "Fetch news on the Indian economy and summarise the key stories.",
                    "Compare sentiment between coverage of OpenAI and Google DeepMind.",
                    "What contradictions exist in how outlets are covering the US economy?",
                ],
                title="",
                retry_btn=None,
                undo_btn=None,
            )

    gr.HTML("""
    <div style="text-align: center; padding: 20px 0 8px; color: #9CA3AF; font-size: 11px; border-top: 1px solid #E5E7EB; margin-top: 24px;">
        Built with LangChain &nbsp;·&nbsp; HuggingFace &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; Groq &nbsp;·&nbsp; Gradio
    </div>
    """)

if __name__ == "__main__":
    demo.launch(show_api=False)
EOFcat > app.py << 'EOF'
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
    sentiment_label = meta["overall_sentiment"].title()
    md = f"## {topic}\n\n"
    md += f"**{analysis['total_articles']} articles analysed** &nbsp;·&nbsp; **{sentiment_label} sentiment**\n\n"
    md += f"> {meta['dominant_narrative']}\n\n"
    if meta.get("contradictions"):
        md += "**Contradictions detected:**\n"
        for c in meta["contradictions"]:
            md += f"- {c}\n"
        md += "\n"
    md += "---\n"
    for k, v in analysis["clusters"].items():
        md += f"\n### Cluster {int(k)+1} &mdash; {v['tone'].title()}\n"
        md += f"{v['narrative']}\n\n"
        md += f"**Sentiment:** {v['sentiment']} &nbsp;·&nbsp; **Articles:** {v['size']}\n\n"
        if v.get("key_claims"):
            md += "**Key claims:**\n"
            for c in v["key_claims"]:
                md += f"- {c}\n"
    return md


def analyze_topic(topic, days_back):
    if not topic.strip():
        return (
            "Please enter a topic.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    articles = fetch_articles(topic.strip(), days_back=int(days_back))
    if not articles:
        return (
            f"No articles found for '{topic}'. Try a broader term or extend the date range.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    embeddings = embed_and_store(articles)
    articles = cluster_articles(articles, embeddings)
    analysis = analyze_with_llm(articles, topic)
    summary_md = build_summary(analysis, topic)

    # Sentiment chart
    cluster_labels = [f"Cluster {int(k)+1}" for k in analysis["clusters"].keys()]
    scores = [v["sentiment_score"] for v in analysis["clusters"].values()]
    fig_s = go.Figure(go.Bar(
        x=cluster_labels, y=scores,
        marker_color=[sentiment_color(s) for s in scores],
        text=[f"{s:+.2f}" for s in scores],
        textposition="outside",
        width=0.45,
    ))
    fig_s.update_layout(
        title=dict(text="Sentiment per Cluster", font=dict(size=13, color="#111827")),
        yaxis=dict(range=[-1.4, 1.4], gridcolor="#E5E7EB", title="Score", zeroline=False),
        xaxis=dict(gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=300,
        margin=dict(l=40, r=40, t=44, b=40),
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
    )
    fig_s.add_hline(y=0, line_dash="dot", line_color="#D1D5DB", line_width=1)

    # Timeline chart
    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    fig_t = px.area(daily, x="date", y="count",
                    title="Article Volume Over Time",
                    color_discrete_sequence=["#2563EB"])
    fig_t.update_traces(line=dict(width=2, color="#2563EB"),
                        fillcolor="rgba(37,99,235,0.08)")
    fig_t.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=300,
        margin=dict(l=40, r=40, t=44, b=40),
        yaxis=dict(title="Articles", gridcolor="#E5E7EB", zeroline=False),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        title=dict(font=dict(size=13, color="#111827")),
    )

    table_rows = [
        [a["title"][:85], a["source"], a["published_at"][:10], f"Cluster {a['cluster']+1}"]
        for a in articles
    ]

    return (
        summary_md,
        gr.update(value=fig_s, visible=True),
        gr.update(value=fig_t, visible=True),
        gr.update(value=table_rows, visible=True),
        gr.update(visible=True),
    )


CSS = """
/* ── Base ── */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Input & button ── */
.gr-button-primary {
    background: #2563EB !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}
.gr-button-primary:hover { background: #1D4ED8 !important; }

/* ── Tab labels ── */
.tab-nav button {
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 8px 0 16px 0;
}

footer { display: none !important; }
"""

with gr.Blocks(
    title="TrendPulse",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=["Inter", "ui-sans-serif", "sans-serif"],
    ),
) as demo:

    gr.HTML("""
    <div style="padding: 24px 0 4px 0; border-bottom: 1px solid #E5E7EB; margin-bottom: 20px;">
        <div style="font-size: 22px; font-weight: 700; color: #111827; letter-spacing: -0.4px;">
            TrendPulse
        </div>
        <div style="font-size: 13px; color: #6B7280; margin-top: 4px;">
            Track narratives, detect sentiment shifts, and surface contradictions across news sources.
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Analyse ──────────────────────────────────────────
        with gr.Tab("Analyse Topic"):

            with gr.Row(equal_height=True):
                topic_input = gr.Textbox(
                    label="Topic",
                    placeholder="e.g. OpenAI  /  Indian economy  /  climate policy  /  Fed rate cuts",
                    scale=5,
                )
                days_slider = gr.Slider(
                    minimum=1, maximum=30, value=7, step=1,
                    label="Days back", scale=1,
                )

            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")

            # Results — hidden until analysis runs
            summary_output = gr.Markdown(visible=True)

            with gr.Row(visible=False) as charts_row:
                sentiment_plot = gr.Plot(show_label=False, visible=False)
                timeline_plot  = gr.Plot(show_label=False, visible=False)

            articles_table = gr.Dataframe(
                headers=["Title", "Source", "Date", "Cluster"],
                label="Articles",
                wrap=True,
                row_count=10,
                visible=False,
            )

            analyse_btn.click(
                fn=analyze_topic,
                inputs=[topic_input, days_slider],
                outputs=[
                    summary_output,
                    sentiment_plot,
                    timeline_plot,
                    articles_table,
                    charts_row,
                ],
            )

        # ── Tab 2: Chat ──────────────────────────────────────────────
        with gr.Tab("Chat with Agent"):
            gr.HTML("""
            <p style="font-size: 13px; color: #6B7280; margin: 12px 0 8px 0;">
                Ask the agent to fetch, analyse, or compare any topic. It remembers your conversation context.
            </p>
            """)
            gr.ChatInterface(
                fn=chat,
                examples=[
                    "What are the dominant narratives around AI regulation right now?",
                    "Fetch news on the Indian economy and summarise the key stories.",
                    "Compare sentiment between coverage of OpenAI and Google DeepMind.",
                    "What contradictions exist in how outlets are covering the US economy?",
                ],
                title="",
                retry_btn=None,
                undo_btn=None,
            )

    gr.HTML("""
    <div style="text-align: center; padding: 20px 0 8px; color: #9CA3AF; font-size: 11px; border-top: 1px solid #E5E7EB; margin-top: 24px;">
        Built with LangChain &nbsp;·&nbsp; HuggingFace &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; Groq &nbsp;·&nbsp; Gradio
    </div>
    """)

if __name__ == "__main__":
    demo.launch(show_api=False)
