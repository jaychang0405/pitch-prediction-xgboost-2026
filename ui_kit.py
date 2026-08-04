# -*- coding: utf-8 -*-
"""共用 UI 設計系統：CSS 主題、卡片元件、圖表元件。
供 home.py 與 pages/*.py 共同 import 使用，避免樣式與圖表邏輯散落各頁。
"""
import random
import streamlit as st
import plotly.graph_objects as go

# ==========================================
# 色彩常數（沿用 .streamlit/config.toml 的深色戰情室主題）
# ==========================================
ACCENT_BLUE = "#42a5f5"
ACCENT_PURPLE = "#8e6bff"
ACCENT_GREEN = "#3ddc97"
ACCENT_AMBER = "#ffb347"
ACCENT_RED = "#ff5c5c"
CARD_BG = "rgba(255, 255, 255, 0.035)"
CARD_BORDER = "rgba(255, 255, 255, 0.09)"


def inject_theme():
    """注入全站共用 CSS：字體、卡片、按鈕、標題漸層等。"""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Sans+TC:wght@400;500;700;900&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Noto Sans TC', 'Inter', sans-serif;
        }}

        /* ---- 捲軸美化 ---- */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(66,165,245,0.35); border-radius: 8px; }}

        /* ---- 通用卡片 ---- */
        .uikit-card {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }}
        .uikit-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(66,165,245,0.55);
            box-shadow: 0 10px 28px rgba(0,0,0,0.35);
        }}

        /* ---- Hero 區塊 ---- */
        .uikit-hero {{
            background: linear-gradient(120deg, rgba(66,165,245,0.16) 0%, rgba(142,107,255,0.14) 100%);
            border: 1px solid rgba(66,165,245,0.25);
            border-radius: 20px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.4rem;
        }}
        .uikit-hero-title {{
            font-size: 2.1rem;
            font-weight: 900;
            margin: 0;
            background: linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_PURPLE});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .uikit-hero-subtitle {{
            color: rgba(255,255,255,0.75);
            font-size: 1.02rem;
            margin-top: 0.35rem;
        }}

        /* ---- 統計小卡 ---- */
        .uikit-stat {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            text-align: left;
        }}
        .uikit-stat-icon {{ font-size: 1.4rem; }}
        .uikit-stat-value {{ font-size: 1.5rem; font-weight: 800; margin: 0.15rem 0; }}
        .uikit-stat-label {{ font-size: 0.82rem; color: rgba(255,255,255,0.6); }}
        .uikit-stat-delta {{ font-size: 0.78rem; font-weight: 600; }}

        /* ---- 機率長條 ---- */
        .uikit-bar-row {{ margin-bottom: 0.65rem; }}
        .uikit-bar-label {{
            display: flex; justify-content: space-between;
            font-size: 0.9rem; margin-bottom: 0.25rem; color: rgba(255,255,255,0.85);
        }}
        .uikit-bar-track {{
            background: rgba(255,255,255,0.07);
            border-radius: 8px;
            height: 12px;
            overflow: hidden;
        }}
        .uikit-bar-fill {{
            height: 100%;
            border-radius: 8px;
            animation: uikit-grow 0.7s ease-out;
        }}
        @keyframes uikit-grow {{
            from {{ width: 0; }}
        }}

        /* ---- 側邊欄狀態卡 ---- */
        .uikit-status-row {{
            display: flex; align-items: center; gap: 0.5rem;
            font-size: 0.85rem; padding: 0.15rem 0;
        }}

        /* ---- Streamlit 按鈕強化 ---- */
        div[data-testid="stButton"] > button {{
            border-radius: 12px;
            font-weight: 700;
            border: 1px solid rgba(66,165,245,0.4);
            transition: all 0.15s ease;
        }}
        div[data-testid="stButton"] > button:hover {{
            border-color: {ACCENT_BLUE};
            box-shadow: 0 0 0 3px rgba(66,165,245,0.18);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_banner(title, subtitle, icon="⚾"):
    st.markdown(
        f"""
        <div class="uikit-hero">
            <div class="uikit-hero-title">{icon} {title}</div>
            <div class="uikit-hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card_row(items):
    """items: list of dict(icon, label, value, delta, delta_color)"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        delta = item.get("delta", "")
        delta_color = item.get("delta_color", ACCENT_GREEN)
        with col:
            st.markdown(
                f"""
                <div class="uikit-stat">
                    <div class="uikit-stat-icon">{item.get('icon', '📊')}</div>
                    <div class="uikit-stat-value">{item['value']}</div>
                    <div class="uikit-stat-label">{item['label']}</div>
                    <div class="uikit-stat-delta" style="color:{delta_color};">{delta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def module_link_card(title, desc, bullets, page_path, icon="⚾", accent=ACCENT_BLUE, link_label="進入模組 →"):
    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(
        f"""
        <div class="uikit-card" style="border-top: 3px solid {accent};">
            <div style="font-size:1.6rem;">{icon}</div>
            <div style="font-size:1.25rem; font-weight:800; margin:0.3rem 0;">{title}</div>
            <div style="color:rgba(255,255,255,0.72); font-size:0.92rem; margin-bottom:0.5rem;">{desc}</div>
            <ul style="color:rgba(255,255,255,0.8); font-size:0.88rem; padding-left:1.1rem;">{bullets_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(page_path, label=link_label, icon="➡️")


def draw_strike_zone_plotly(predicted_pitch_en, prob):
    """九宮格互動落點圖（CPBL / MLB 共用）。"""
    if "Fastball" in predicted_pitch_en:
        target_id = random.choice([2, 5, 4, 6])
    elif "Changeup" in predicted_pitch_en or "Splitter" in predicted_pitch_en:
        target_id = random.choice([7, 8, 9])
    else:
        target_id = random.choice([1, 3, 7, 9])

    fig = go.Figure()
    x_centers, y_centers, texts, hover_texts, text_colors = [], [], [], [], []

    for i in range(1, 10):
        col = (i - 1) % 3
        row = 2 - ((i - 1) // 3)
        x_centers.append(col + 0.5)
        y_centers.append(row + 0.5)
        is_target = (i == target_id)

        if is_target:
            texts.append(f"<b>{predicted_pitch_en}</b><br>{prob:.1f}%")
            hover_texts.append(f"區域: {i}<br>預測落點: {predicted_pitch_en}")
            text_colors.append("#ffffff")
            fig.add_shape(
                type="rect", x0=col, y0=row, x1=col + 1, y1=row + 1,
                fillcolor="rgba(255, 92, 92, 0.45)",
                line=dict(color=ACCENT_RED, width=3),
                layer="below",
            )
        else:
            texts.append(str(i))
            hover_texts.append(f"區域: {i}<br>非主要落點")
            text_colors.append("rgba(255, 255, 255, 0.3)")
            fig.add_shape(
                type="rect", x0=col, y0=row, x1=col + 1, y1=row + 1,
                fillcolor="rgba(66, 165, 245, 0.06)",
                line=dict(color=ACCENT_BLUE, width=1),
                layer="below",
            )

    fig.add_trace(go.Scatter(
        x=x_centers, y=y_centers, mode="text", text=texts,
        hoverinfo="text", hovertext=hover_texts,
        textfont=dict(color=text_colors, size=15), showlegend=False,
    ))

    fig.update_layout(
        width=350, height=350,
        xaxis=dict(range=[0, 3], showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(range=[0, 3], showgrid=False, zeroline=False, visible=False, fixedrange=True,
                    scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        title=dict(text="AI 預測落點 (Strike Zone)", x=0.5, font=dict(size=16, color="#e0e0e0")),
    )
    return fig


def probability_bars(names, probs, highlight_idx=None):
    """自訂橫向漸層機率條，取代 st.bar_chart。"""
    if highlight_idx is None:
        highlight_idx = max(range(len(probs)), key=lambda i: probs[i])

    rows = []
    for i, (name, prob) in enumerate(zip(names, probs)):
        is_top = (i == highlight_idx)
        gradient = f"linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_PURPLE})" if is_top else "rgba(255,255,255,0.25)"
        width = max(prob, 1.5)
        label = f"{'🎯 ' if is_top else ''}{name}"
        # Single-line HTML per row: Streamlit's markdown parser treats 4+ space
        # indented lines as code blocks, so multi-line indented f-strings here
        # would leak as literal text instead of rendering as HTML.
        rows.append(
            f'<div class="uikit-bar-row"><div class="uikit-bar-label">'
            f'<span>{label}</span><span><b>{prob:.1f}%</b></span></div>'
            f'<div class="uikit-bar-track"><div class="uikit-bar-fill" '
            f'style="width:{width}%; background:{gradient};"></div></div></div>'
        )
    st.markdown(f'<div class="uikit-card">{"".join(rows)}</div>', unsafe_allow_html=True)


def risk_gauge(prob, title="上壘機率 (xOBP)", low_threshold=0.30, high_threshold=0.38):
    """Plotly gauge：0~1 的機率風險儀表，三段色區。"""
    pct = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 34, "color": "#fafafa"}},
        title={"text": title, "font": {"size": 15, "color": "rgba(255,255,255,0.75)"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.4)"},
            "bar": {"color": ACCENT_BLUE, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, low_threshold * 100], "color": "rgba(61,220,151,0.35)"},
                {"range": [low_threshold * 100, high_threshold * 100], "color": "rgba(255,179,71,0.35)"},
                {"range": [high_threshold * 100, 100], "color": "rgba(255,92,92,0.35)"},
            ],
            "threshold": {
                "line": {"color": ACCENT_RED, "width": 3},
                "thickness": 0.85,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa"),
    )
    return fig


def sidebar_status(logo_path, brand, model_status: dict):
    """側邊欄品牌 Logo + 模型載入狀態卡。"""
    try:
        st.logo(logo_path)
    except Exception:
        pass
    st.sidebar.markdown(f"### {brand}")
    st.sidebar.markdown('<div class="uikit-card">', unsafe_allow_html=True)
    st.sidebar.markdown("**🔌 系統狀態**")
    for label, ok in model_status.items():
        dot = "🟢" if ok else "🔴"
        st.sidebar.markdown(
            f'<div class="uikit-status-row">{dot} {label}</div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
