import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.fetcher import fetch_articles
from src.embedder import embed_and_store
from src.analyzer import cluster_articles, analyze_with_llm
from src.agent import chat as agent_chat

EXAMPLES = ["OpenAI","Indian economy","Climate policy","Fed rate cuts","India-Pakistan","Nvidia","US elections","Bitcoin","Gaza conflict","AI regulation"]

def sentiment_color(score):
    if score > 0.1: return "#16A34A"
    if score < -0.1: return "#DC2626"
    return "#6B7280"

def build_summary(analysis, topic):
    meta = analysis["meta"]
    md = f"## {topic}\n\n**{analysis['total_articles']} articles analysed** &nbsp;·&nbsp; **{meta['overall_sentiment'].title()} sentiment**\n\n> {meta['dominant_narrative']}\n\n"
    if meta.get("contradictions"):
        md += "**Contradictions detected:**\n" + "".join([f"- {c}\n" for c in meta["contradictions"]]) + "\n"
    md += "---\n"
    for k, v in analysis["clusters"].items():
        md += f"\n### Cluster {int(k)+1} &mdash; {v['tone'].title()}\n{v['narrative']}\n\n**Sentiment:** {v['sentiment']} &nbsp;·&nbsp; **Articles:** {v['size']}\n\n"
        if v.get("key_claims"):
            md += "**Key claims:**\n" + "".join([f"- {c}\n" for c in v["key_claims"]])
    return md

def format_recent(recent):
    if not recent:
        return '<p style="font-size:12px;color:#CBD5E1;font-style:italic;margin:0;">No recent searches yet.</p>'
    return "".join([f'<div style="font-size:13px;color:#64748B;padding:6px 10px;border-radius:6px;background:#F8FAFC;border:1px solid #E2E8F0;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{r}</div>' for r in reversed(recent[-6:])])

def analyze_topic(topic, days_back, recent):
    if not topic.strip():
        return "", gr.update(visible=False), None, None, [], format_recent(recent), recent
    articles = fetch_articles(topic.strip(), days_back=int(days_back))
    if not articles:
        return f"No articles found for **'{topic}'**. Try a broader term.", gr.update(visible=False), None, None, [], format_recent(recent), recent
    embeddings = embed_and_store(articles)
    articles = cluster_articles(articles, embeddings)
    analysis = analyze_with_llm(articles, topic)
    cluster_labels = [f"Cluster {int(k)+1}" for k in analysis["clusters"].keys()]
    scores = [v["sentiment_score"] for v in analysis["clusters"].values()]
    fig_s = go.Figure(go.Bar(x=cluster_labels, y=scores, marker_color=[sentiment_color(s) for s in scores], text=[f"{s:+.2f}" for s in scores], textposition="outside", width=0.45))
    fig_s.update_layout(title=dict(text="Sentiment per Cluster", font=dict(size=13, color="#1E293B")), yaxis=dict(range=[-1.4,1.4], gridcolor="#F1F5F9", title="Score", zeroline=False, color="#64748B"), xaxis=dict(gridcolor="#F1F5F9", color="#64748B"), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=280, margin=dict(l=40,r=20,t=44,b=40), font=dict(family="Inter, sans-serif", size=12, color="#374151"))
    fig_s.add_hline(y=0, line_dash="dot", line_color="#E2E8F0", line_width=1.5)
    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    fig_t = px.area(daily, x="date", y="count", title="Article Volume", color_discrete_sequence=["#3B82F6"])
    fig_t.update_traces(line=dict(width=2, color="#3B82F6"), fillcolor="rgba(59,130,246,0.07)")
    fig_t.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=280, margin=dict(l=40,r=20,t=44,b=40), yaxis=dict(title="Articles", gridcolor="#F1F5F9", zeroline=False, color="#64748B"), xaxis=dict(title="", gridcolor="#F1F5F9", color="#64748B"), font=dict(family="Inter, sans-serif", size=12, color="#374151"), title=dict(font=dict(size=13, color="#1E293B")))
    table_rows = [[a["title"][:85], a["source"], a["published_at"][:10], f"Cluster {a['cluster']+1}"] for a in articles]
    updated = recent + [topic.strip()]
    return build_summary(analysis, topic), gr.update(visible=True), fig_s, fig_t, table_rows, format_recent(updated), updated

CSS = """
html, body, .gradio-container, .main, .wrap { background: #F8FAFC !important; color: #1E293B !important; }
input, textarea, .block { background: #FFFFFF !important; color: #1E293B !important; border-color: #E2E8F0 !important; }
label, .label-wrap span { color: #475569 !important; font-size: 13px !important; font-weight: 500 !important; }
.gradio-container { max-width: 1300px !important; margin: 0 auto !important; font-family: 'Inter', sans-serif !important; padding: 0 !important; }
.tab-nav { background: #FFFFFF !important; border-bottom: 1px solid #E2E8F0 !important; padding: 0 24px !important; }
.tab-nav button { font-size: 13px !important; font-weight: 600 !important; color: #94A3B8 !important; padding: 14px 16px !important; background: transparent !important; }
.tab-nav button.selected { color: #2563EB !important; border-bottom: 2px solid #2563EB !important; }
#sidebar { background: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; padding: 28px 18px !important; min-height: calc(100vh - 52px); }
#main-area { background: #F8FAFC !important; padding: 32px 40px !important; }
#search-box { background: #FFFFFF !important; border: 1.5px solid #E2E8F0 !important; border-radius: 14px !important; padding: 20px 24px 16px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important; margin-bottom: 20px !important; }
#analyse-btn > div > button, #analyse-btn button { background: linear-gradient(135deg, #2563EB, #1D4ED8) !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; font-size: 14px !important; color: #FFFFFF !important; height: 46px !important; width: 100% !important; box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important; }
#analyse-btn > div > button:hover, #analyse-btn button:hover { background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important; transform: translateY(-1px) !important; }
#results-card { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 14px !important; padding: 28px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important; }
.gr-dataframe th, thead th { background: #F8FAFC !important; color: #475569 !important; font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
.gr-dataframe td, tbody td { font-size: 13px !important; color: #374151 !important; }
#sidebar button { background: #F8FAFC !important; border: 1px solid #E2E8F0 !important; color: #374151 !important; text-align: left !important; border-radius: 8px !important; font-size: 13px !important; padding: 8px 12px !important; margin-bottom: 6px !important; width: 100% !important; box-shadow: none !important; justify-content: flex-start !important; font-weight: 400 !important; }
#sidebar button:hover { background: #EEF2FF !important; border-color: #BFDBFE !important; color: #2563EB !important; }
.prose h2, .md h2 { color: #0F172A !important; font-size: 20px !important; font-weight: 700 !important; }
.prose h3, .md h3 { color: #1E293B !important; font-size: 15px !important; font-weight: 600 !important; }
.prose blockquote, .md blockquote { border-left: 3px solid #3B82F6 !important; background: #EFF6FF !important; padding: 12px 16px !important; border-radius: 0 8px 8px 0 !important; color: #1E40AF !important; }
input[type=range] { accent-color: #2563EB !important; }
#chat-area { background: #F8FAFC !important; padding: 32px 40px !important; }
footer { display: none !important; }
"""

HERO = """<div style="background:linear-gradient(135deg,#EFF6FF 0%,#F8FAFC 50%,#F0FDF4 100%);border:1px solid #E2E8F0;border-radius:14px;padding:32px 36px;margin-bottom:24px;position:relative;overflow:hidden;">
<div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;border-radius:50%;background:radial-gradient(circle,rgba(59,130,246,0.08) 0%,transparent 70%);"></div>
<div style="font-size:26px;font-weight:800;color:#0F172A;letter-spacing:-0.6px;margin-bottom:8px;">TrendPulse</div>
<div style="font-size:14px;color:#64748B;line-height:1.6;max-width:480px;">Enter any topic to analyse live news — track narratives, detect sentiment shifts, and surface contradictions across sources.</div>
<div style="display:flex;gap:20px;margin-top:18px;">
<div style="display:flex;align-items:center;gap:7px;font-size:12.5px;color:#475569;"><div style="width:8px;height:8px;border-radius:50%;background:#16A34A;"></div>Narrative clustering</div>
<div style="display:flex;align-items:center;gap:7px;font-size:12.5px;color:#475569;"><div style="width:8px;height:8px;border-radius:50%;background:#3B82F6;"></div>Sentiment analysis</div>
<div style="display:flex;align-items:center;gap:7px;font-size:12.5px;color:#475569;"><div style="width:8px;height:8px;border-radius:50%;background:#F59E0B;"></div>Contradiction detection</div>
</div></div>"""

with gr.Blocks(title="TrendPulse", css=CSS, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate", font=["Inter","ui-sans-serif","sans-serif"])) as demo:
    recent_state = gr.State([])
    with gr.Tabs():
        with gr.Tab("Analyse Topic"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=220, elem_id="sidebar"):
                    gr.HTML('<div style="font-size:18px;font-weight:700;color:#0F172A;letter-spacing:-0.3px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid #E2E8F0;">TrendPulse</div><div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin-bottom:10px;">Try these topics</div>')
                    example_btns = [gr.Button(ex, size="sm") for ex in EXAMPLES]
                    gr.HTML('<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin:24px 0 10px 0;">Recent searches</div>')
                    recent_html = gr.HTML('<p style="font-size:12px;color:#CBD5E1;font-style:italic;margin:0;">No recent searches yet.</p>')
                with gr.Column(scale=4, elem_id="main-area"):
                    gr.HTML(HERO)
                    with gr.Group(elem_id="search-box"):
                        with gr.Row(equal_height=True):
                            topic_input = gr.Textbox(label="Topic", placeholder="e.g. OpenAI, Indian economy, climate policy, Bitcoin ...", scale=5)
                            days_slider = gr.Slider(minimum=1, maximum=30, value=7, step=1, label="Days back", scale=1)
                        with gr.Row(elem_id="analyse-btn"):
                            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")
                    summary_output = gr.Markdown()
                    with gr.Column(visible=False, elem_id="results-card") as results_col:
                        with gr.Row():
                            sentiment_plot = gr.Plot(show_label=False)
                            timeline_plot  = gr.Plot(show_label=False)
                        gr.HTML('<div style="height:12px;"></div>')
                        articles_table = gr.Dataframe(headers=["Title","Source","Date","Cluster"], label="All articles", wrap=True, row_count=8)
            for btn in example_btns:
                btn.click(fn=lambda x=btn.value: x, outputs=topic_input)
            analyse_btn.click(fn=analyze_topic, inputs=[topic_input, days_slider, recent_state], outputs=[summary_output, results_col, sentiment_plot, timeline_plot, articles_table, recent_html, recent_state])

        with gr.Tab("Chat with Agent"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=220, elem_id="sidebar"):
                    gr.HTML("""<div style="font-size:18px;font-weight:700;color:#0F172A;letter-spacing:-0.3px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid #E2E8F0;">TrendPulse</div>
                    <div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin-bottom:12px;">What the agent can do</div>
                    <div style="display:flex;flex-direction:column;gap:6px;margin-bottom:24px;">
                    <div style="font-size:13px;color:#475569;padding:8px 12px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;">Analyse any topic live</div>
                    <div style="font-size:13px;color:#475569;padding:8px 12px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;">Compare two topics</div>
                    <div style="font-size:13px;color:#475569;padding:8px 12px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;">Find contradictions</div>
                    <div style="font-size:13px;color:#475569;padding:8px 12px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;">Summarise coverage</div>
                    <div style="font-size:13px;color:#475569;padding:8px 12px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;">Search stored articles</div>
                    </div>
                    <div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin-bottom:12px;">Example prompts</div>
                    <div style="font-size:12.5px;color:#94A3B8;line-height:2.4;font-style:italic;">
                    "Narratives around AI regulation?"<br>"Compare OpenAI vs DeepMind"<br>"Contradictions in US economy?"<br>"Summarise Fed rate cut news"</div>""")
                with gr.Column(scale=4, elem_id="chat-area"):
                    gr.ChatInterface(fn=agent_chat, title="")

    gr.HTML('<div style="text-align:center;padding:16px 0 10px;color:#CBD5E1;font-size:11px;border-top:1px solid #F1F5F9;margin-top:8px;background:#F8FAFC;">LangChain &nbsp;·&nbsp; HuggingFace Transformers &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; Groq LLaMA &nbsp;·&nbsp; Gradio</div>')

if __name__ == "__main__":
    demo.launch(show_api=False)
