# -*- coding: utf-8 -*-
import streamlit as st

# ==========================================
# 1. 頁面基本設定 (開啟寬螢幕模式 layout="wide")
# ==========================================
st.set_page_config(
    page_title="棒球動態決策支援中心",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Hero 區塊 (大標題與歡迎詞)
# ==========================================
st.title("⚾ 棒球動態決策支援中心")
st.markdown("#### 即時戰術與上壘風險預測系統 (Powered by XGBoost)")
st.markdown("歡迎來到資料科學與棒球賽事的交會點。本系統結合歷史投打對決數據，為您提供即時的情境預測。")
st.divider() # 水平分隔線

# ==========================================
# 3. 數據概覽面板 (Metrics)
# ==========================================
st.markdown("### 📊 系統運行狀態")
# 建立 4 個等寬欄位
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(label="核心預測引擎", value="XGBoost", delta="Active", delta_color="normal")
with m2:
    st.metric(label="CPBL 模型準確率", value="72.4%", delta="+1.2%", delta_color="normal")
with m3:
    st.metric(label="MLB 模型準確率", value="76.1%", delta="+2.5%", delta_color="normal")
with m4:
    st.metric(label="系統響應時間", value="0.15s", delta="-0.02s", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True) # 增加一點垂直空白

# ==========================================
# 4. 核心功能介紹卡片 (Columns & Info Boxes)
# ==========================================
st.markdown("### 🌟 核心預測模組")
col1, col2 = st.columns(2)

with col1:
    # 使用 info 創造淺藍色卡片感
    st.info("#### CPBL 中職預測系統")
    st.write("針對中華職棒本土與洋將特化的機器學習模型。")
    st.write("- **🎯 球種預測**：根據局數、球數與前一球軌跡，預測投手下一球配球。")
    st.write("- **🏃‍♂️ OBP 分析**：結合打者歷史上壘率與當前壘包狀態，評估高張力打席。")
    st.caption("👈 請於左側選單點擊進入 CPBL 模組")

with col2:
    # 使用 success 創造淺綠色卡片感
    st.success("#### MLB 大聯盟預測系統")
    st.write("基於 Statcast 進階數據的大聯盟級別戰術分析。")
    st.write("- **🔥 頂級對決**：包含大聯盟頂級投手的球路特徵解析。")
    st.write("- **📈 進階特徵**：整合更豐富的賽事數據與情境參數。")
    st.caption("👈 請於左側選單點擊進入 MLB 模組")

st.divider()

# ==========================================
# 5. 使用說明 (折疊區塊 Expander)
# ==========================================
with st.expander("📖 系統使用指南與注意事項 (點擊展開)"):
    st.markdown("""
    * **系統解鎖**：進入預測模組後，請輸入通關密碼解鎖預測功能。
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
        <small>© 2026 Baseball Analytics Project | Powered by Streamlit & XGBoost</small>
    </div>
    """, 
    unsafe_allow_html=True
)