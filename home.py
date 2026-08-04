# -*- coding: utf-8 -*-
import streamlit as st
import ui_kit

# ==========================================
# 1. 頁面基本設定 (開啟寬螢幕模式 layout="wide")
# ==========================================
st.set_page_config(
    page_title="棒球動態決策支援系統",
    page_icon="data_cpbl/tennis-ball.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

ui_kit.inject_theme()
ui_kit.sidebar_status(
    logo_path="data_cpbl/cpbl_icon.png",
    brand="⚾ Dynamic Decision System",
    model_status={
        "首頁": True,
        "CPBL 模組": True,
        "MLB 模組": True,
        "3D 共軌實驗室": True,
    },
)

# ==========================================
# 2. Hero 區塊 (大標題與歡迎詞)
# ==========================================
ui_kit.hero_banner(
    "棒球動態決策支援系統",
    "即時球種與上壘風險預測系統 (Powered by XGBoost) — 歡迎來到資料科學與棒球賽事的交會點。本系統結合歷史投打對決數據，為您提供即時的情境預測。",
    icon="⚾",
)

# ==========================================
# 3. 數據概覽面板 (Metrics)
# ==========================================
st.markdown("### 📈 系統運行狀態")
ui_kit.stat_card_row([
    {"icon": "🧠", "label": "核心預測引擎", "value": "XGBoost", "delta": "● Active", "delta_color": ui_kit.ACCENT_GREEN},
    {"icon": "🇹🇼", "label": "CPBL 模型準確率", "value": "72.4%", "delta": "▲ +1.2%", "delta_color": ui_kit.ACCENT_GREEN},
    {"icon": "🇺🇸", "label": "MLB 模型準確率", "value": "76.1%", "delta": "▲ +2.5%", "delta_color": ui_kit.ACCENT_GREEN},
    {"icon": "⚡", "label": "系統響應時間", "value": "0.15s", "delta": "▼ -0.02s", "delta_color": ui_kit.ACCENT_BLUE},
])

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 4. 核心功能介紹卡片
# ==========================================
st.markdown("### 🧩 核心預測模組")
col1, col2, col3 = st.columns(3)

with col1:
    ui_kit.module_link_card(
        title="CPBL 中職預測系統",
        desc="針對中華職棒本土與洋將特化的機器學習模型。",
        bullets=[
            "球種預測：根據局數、球數與前一球軌跡，預測投手下一球配球。",
            "OBP 分析：結合打者歷史上壘率與當前壘包狀態，評估高張力打席。",
        ],
        page_path="pages/1_CPBL_app.py",
        icon="🇹🇼",
        accent=ui_kit.ACCENT_BLUE,
    )

with col2:
    ui_kit.module_link_card(
        title="MLB 大聯盟預測系統",
        desc="基於 Statcast 進階數據的 MLB 戰術分析。",
        bullets=[
            "頂級對決：包含大聯盟頂級投手的球路特徵解析。",
            "進階特徵：整合更豐富的賽事數據與情境參數。",
        ],
        page_path="pages/2_MLB_app.py",
        icon="🇺🇸",
        accent=ui_kit.ACCENT_PURPLE,
    )

with col3:
    ui_kit.module_link_card(
        title="3D 共軌效應實驗室",
        desc="互動式 3D 視角，體驗打者眼中直球與變化球的視覺欺騙。",
        bullets=[
            "打者視角：親自旋轉、縮放球路軌跡 3D 圖。",
            "模型解密：理解為何 AI 會把直球誤判成變化球。",
        ],
        page_path="pages/3_3D_Tunneling_Lab.py",
        icon="🌀",
        accent=ui_kit.ACCENT_AMBER,
    )

st.divider()

# ==========================================
# 5. 使用說明 (折疊區塊 Expander)
# ==========================================
with st.expander("📖 系統使用指南與注意事項 (點擊展開)"):
    st.markdown("""
    * **情境設定**：請確實依照實際比賽情況（如好壞球、出局數、壘上情形）進行設定，以獲得最準確的預測機率。
    * **動態數據**：系統會根據您選擇的球員，自動至背景資料庫 (`_db.csv`) 撈取歷史數據作為模型特徵。
    * **免責聲明**：本系統預測結果僅供棒球數據研究與觀賽娛樂參考，不代表絕對賽事結果。
    """)

# ==========================================
# 6. 頁尾版權宣告
# ==========================================
st.markdown(
    """
    <div style='text-align: center; color: gray; padding-top: 20px;'>
        <small>⚾ © 2026 Baseball Analytics Project | Powered by Streamlit &amp; XGBoost</small>
    </div>
    """,
    unsafe_allow_html=True
)
