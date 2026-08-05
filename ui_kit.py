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

# 舊常數保留給還沒 import palette() 的地方用，對應深色模式的預設值。
CARD_BG = "rgba(255, 255, 255, 0.035)"
CARD_BORDER = "rgba(255, 255, 255, 0.09)"

LANGUAGES = {"繁體中文": "zh", "English": "en", "日本語": "ja"}
LANG_SWITCH_KEY = "lang_switch"

_DARK_PALETTE = {
    "text": "#fafafa",
    "text_soft": "rgba(255,255,255,0.75)",
    "text_muted": "rgba(255,255,255,0.6)",
    "card_bg": "rgba(255,255,255,0.035)",
    "card_border": "rgba(255,255,255,0.09)",
    "card_shadow": "rgba(0,0,0,0.35)",
    "hero_bg": "linear-gradient(120deg, rgba(66,165,245,0.16) 0%, rgba(142,107,255,0.14) 100%)",
    "hero_border": "rgba(66,165,245,0.25)",
    "track_bg": "rgba(255,255,255,0.07)",
    "bar_off": "rgba(255,255,255,0.25)",
    "row_border": "rgba(255,255,255,0.08)",
    "chart_text": "#e0e0e0",
    "chart_strong": "#ffffff",
    "chart_muted": "rgba(255,255,255,0.4)",
    "chart_faint": "rgba(255,255,255,0.3)",
}
_LIGHT_PALETTE = {
    "text": "#181c24",
    "text_soft": "rgba(10,15,25,0.72)",
    "text_muted": "rgba(10,15,25,0.55)",
    "card_bg": "rgba(10,15,30,0.03)",
    "card_border": "rgba(10,15,30,0.12)",
    "card_shadow": "rgba(20,30,60,0.14)",
    "hero_bg": "linear-gradient(120deg, rgba(66,165,245,0.10) 0%, rgba(142,107,255,0.09) 100%)",
    "hero_border": "rgba(66,165,245,0.30)",
    "track_bg": "rgba(10,15,30,0.08)",
    "bar_off": "rgba(10,15,30,0.18)",
    "row_border": "rgba(10,15,30,0.10)",
    "chart_text": "#20242c",
    "chart_strong": "#0d0f14",
    "chart_muted": "rgba(20,24,32,0.5)",
    "chart_faint": "rgba(20,24,32,0.35)",
}


def theme_type():
    """回傳目前 Streamlit 使用者實際套用的主題："light" 或 "dark"。
    讀取失敗（例如極舊版本沒有 st.context.theme）時預設 dark。"""
    try:
        return st.context.theme.type or "dark"
    except Exception:
        return "dark"


def palette():
    """回傳目前主題對應的色彩字典，供 ui_kit 內部與各頁面的自訂 HTML/Plotly 共用。"""
    return _LIGHT_PALETTE if theme_type() == "light" else _DARK_PALETTE


def inject_theme():
    """注入全站共用 CSS：字體、卡片、按鈕、標題漸層等。會依 Streamlit 目前的
    Light/Dark 主題（設定 > Choose app theme）自動套用對應色盤。"""
    p = palette()
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
            background: {p['card_bg']};
            border: 1px solid {p['card_border']};
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }}
        .uikit-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(66,165,245,0.55);
            box-shadow: 0 10px 28px {p['card_shadow']};
        }}

        /* ---- Hero 區塊 ---- */
        .uikit-hero {{
            background: {p['hero_bg']};
            border: 1px solid {p['hero_border']};
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
            color: {p['text_soft']};
            font-size: 1.02rem;
            margin-top: 0.35rem;
        }}

        /* ---- 統計小卡 ---- */
        .uikit-stat {{
            background: {p['card_bg']};
            border: 1px solid {p['card_border']};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            text-align: left;
        }}
        .uikit-stat-icon {{ font-size: 1.4rem; }}
        .uikit-stat-value {{ font-size: 1.5rem; font-weight: 800; margin: 0.15rem 0; }}
        .uikit-stat-label {{ font-size: 0.82rem; color: {p['text_muted']}; }}
        .uikit-stat-delta {{ font-size: 0.78rem; font-weight: 600; }}

        /* ---- 球員卡：圓形大頭照 ---- */
        .uikit-player-card {{
            text-align: center;
            padding: 0.6rem 0.3rem 0.2rem;
        }}
        .uikit-player-avatar {{
            width: 84px;
            height: 84px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid rgba(66,165,245,0.45);
            box-shadow: 0 4px 14px {p['card_shadow']};
        }}
        .uikit-player-name {{
            font-weight: 700;
            font-size: 0.9rem;
            margin-top: 0.5rem;
            line-height: 1.25;
            min-height: 2.3em;
        }}

        /* ---- 機率長條 ---- */
        .uikit-bar-row {{ margin-bottom: 0.65rem; }}
        .uikit-bar-label {{
            display: flex; justify-content: space-between;
            font-size: 0.9rem; margin-bottom: 0.25rem; color: {p['text_soft']};
        }}
        .uikit-bar-track {{
            background: {p['track_bg']};
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

        /* ---- 頂部導覽列 (st.navigation position="top") ---- */
        [data-testid="stHeader"] {{
            background: linear-gradient(90deg, #0a1a35 0%, #123467 100%) !important;
            border-bottom: 1px solid rgba(66,165,245,0.32);
            box-shadow: 0 4px 18px rgba(0,0,0,0.4);
        }}
        [data-testid="stHeaderLogo"] img {{
            filter: drop-shadow(0 0 6px rgba(66,165,245,0.6));
        }}
        [data-testid="stTopNavLink"] {{
            color: rgba(255,255,255,0.78) !important;
            font-weight: 700 !important;
            border-radius: 999px !important;
            padding: 0.35rem 1.1rem !important;
            margin: 0 0.15rem !important;
            transition: all 0.15s ease;
        }}
        [data-testid="stTopNavLink"]:hover {{
            color: #ffffff !important;
            background: rgba(255,255,255,0.10) !important;
        }}
        [data-testid="stTopNavLink"][aria-current="page"] {{
            color: #ffffff !important;
            background: linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_PURPLE}) !important;
            box-shadow: 0 2px 10px rgba(66,165,245,0.45);
        }}

        /* ---- 語言切換器：以 fixed 定位疊到頂部導覽列右側 ---- */
        [data-testid="stAppDeployButton"] {{
            display: none !important;
        }}
        div.st-key-{LANG_SWITCH_KEY} {{
            position: fixed;
            top: 10px;
            right: 3.2rem;
            width: 148px;
            z-index: 999999;
        }}
        div.st-key-{LANG_SWITCH_KEY} div[data-baseweb="select"] > div {{
            background: rgba(255,255,255,0.08) !important;
            border-color: rgba(255,255,255,0.25) !important;
            color: #ffffff !important;
            min-height: 40px !important;
        }}
        div.st-key-{LANG_SWITCH_KEY} svg {{
            fill: #ffffff !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def language_switcher(default="繁體中文"):
    """右上角固定定位的語言下拉選單，疊在頂部導覽列上。回傳語言代碼 (zh/en/ja)。"""
    options = list(LANGUAGES.keys())
    with st.container(key=LANG_SWITCH_KEY):
        choice = st.selectbox(
            "Language",
            options,
            index=options.index(default),
            label_visibility="collapsed",
        )
    return LANGUAGES[choice]


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
    p = palette()
    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(
        f"""
        <div class="uikit-card" style="border-top: 3px solid {accent};">
            <div style="font-size:1.6rem;">{icon}</div>
            <div style="font-size:1.25rem; font-weight:800; margin:0.3rem 0;">{title}</div>
            <div style="color:{p['text_soft']}; font-size:0.92rem; margin-bottom:0.5rem;">{desc}</div>
            <ul style="color:{p['text_soft']}; font-size:0.88rem; padding-left:1.1rem;">{bullets_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(page_path, label=link_label, icon="➡️")


STRIKE_ZONE_LABELS = {
    "zh": {"title": "AI 預測落點 (Strike Zone)", "area": "區域", "predicted": "預測落點", "non_primary": "非主要落點"},
    "en": {"title": "AI Predicted Location (Strike Zone)", "area": "Zone", "predicted": "Predicted", "non_primary": "Not primary"},
    "ja": {"title": "AI予測コース (Strike Zone)", "area": "ゾーン", "predicted": "予測コース", "non_primary": "非主要コース"},
}


def draw_strike_zone_plotly(predicted_pitch_en, prob, lang="zh"):
    """九宮格互動落點圖（CPBL / MLB 共用）。"""
    p = palette()
    labels = STRIKE_ZONE_LABELS.get(lang, STRIKE_ZONE_LABELS["zh"])
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
            hover_texts.append(f"{labels['area']}: {i}<br>{labels['predicted']}: {predicted_pitch_en}")
            text_colors.append(p["chart_strong"])
            fig.add_shape(
                type="rect", x0=col, y0=row, x1=col + 1, y1=row + 1,
                fillcolor="rgba(255, 92, 92, 0.45)",
                line=dict(color=ACCENT_RED, width=3),
                layer="below",
            )
        else:
            texts.append(str(i))
            hover_texts.append(f"{labels['area']}: {i}<br>{labels['non_primary']}")
            text_colors.append(p["chart_faint"])
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
        title=dict(text=labels["title"], x=0.5, font=dict(size=16, color=p["chart_text"])),
    )
    return fig


def probability_bars(names, probs, highlight_idx=None):
    """自訂橫向漸層機率條，取代 st.bar_chart。"""
    p = palette()
    if highlight_idx is None:
        highlight_idx = max(range(len(probs)), key=lambda i: probs[i])

    rows = []
    for i, (name, prob) in enumerate(zip(names, probs)):
        is_top = (i == highlight_idx)
        gradient = f"linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_PURPLE})" if is_top else p["bar_off"]
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


def segmented_nav(label, options, default, key, format_func=None):
    """包住 st.segmented_control() 後面「使用者取消選取時 fallback 回 default」
    的樣板：segmented_control 允許取消選取而回傳 None，但這裡的導覽列/模式切換
    永遠要有一個選項是「目前選的」，所以統一在這裡補回 default。"""
    kwargs = {"default": default, "label_visibility": "collapsed", "key": key}
    if format_func is not None:
        kwargs["format_func"] = format_func
    selected = st.segmented_control(label, options, **kwargs)
    return default if selected is None else selected


def render_pitch_result(result_header, top_pick_label, second_pick_label, strike_zone_header,
                         best_name, best_prob, second_name, second_prob,
                         chart_names, chart_probs, plot_pitch_key_en, lang):
    """球種預測結果排版（CPBL / MLB 共用）：左欄 metric 卡片 + 機率長條，
    右欄九宮格落點圖。"""
    res_c1, res_c2 = st.columns([1, 1])
    with res_c1:
        st.markdown(f"##### {result_header}")
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label=top_pick_label, value=best_name, delta=f"{best_prob:.1f}%")
        with m2:
            st.metric(label=second_pick_label, value=second_name, delta=f"{second_prob:.1f}%", delta_color="off")
        probability_bars(chart_names, chart_probs)
    with res_c2:
        st.markdown(f"##### {strike_zone_header}")
        fig = draw_strike_zone_plotly(plot_pitch_key_en, best_prob, lang=lang)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_obp_result(metric_label, prob, high_risk_msg, low_risk_msg,
                       gauge_low_threshold=0.30, gauge_high_threshold=0.38,
                       risk_cutoff=0.35, high_risk_severity="warning"):
    """OBP 預測結果排版（CPBL / MLB 共用）：左欄風險儀表，右欄 metric + 高/低
    風險文字。兩聯盟門檻值與嚴重度（error/warning）不同，故皆開放參數化，
    不強行統一數值。"""
    res_c1, res_c2 = st.columns([1, 1])
    with res_c1:
        fig = risk_gauge(float(prob), title=metric_label,
                          low_threshold=gauge_low_threshold, high_threshold=gauge_high_threshold)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with res_c2:
        st.metric(label=metric_label, value=f"{prob:.1%}")
        if prob > risk_cutoff:
            (st.error if high_risk_severity == "error" else st.warning)(high_risk_msg)
        else:
            st.success(low_risk_msg)


def risk_gauge(prob, title="上壘機率 (xOBP)", low_threshold=0.30, high_threshold=0.38):
    """Plotly gauge：0~1 的機率風險儀表，三段色區。"""
    p = palette()
    pct = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 34, "color": p["chart_strong"]}},
        title={"text": title, "font": {"size": 15, "color": p["text_soft"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": p["chart_muted"]},
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
        font=dict(color=p["chart_strong"]),
    )
    return fig
