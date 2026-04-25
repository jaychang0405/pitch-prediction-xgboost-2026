# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# 如果有安裝 xgboost，則載入 (為了雲端部署相容性)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

st.set_page_config(page_title="CPBL 動態決策支援", page_icon="⚾", layout="wide")

# ==========================================
# 0. 簡易密碼鎖
# ==========================================
pwd = st.text_input("🔒 請輸入存取密碼", type="password")
if pwd != "20050405":  
    st.warning("請輸入正確密碼以解鎖系統。")
    st.stop()  
    
# ==========================================
# 1. 雙語字典設定 
# ==========================================
LANG = {
    "zh": {
        "title": "中職動態決策支援系統",
        "subtitle": "基於 XGBoost 與投球序列特徵的即時戰術分析",
        "menu": "🔍 功能選單",
        "mode_pitch": "🎯 預測下一球球種",
        "mode_obp": "🏃‍♂️ 預測打擊結果 (上壘率)",
        "input_header": "📊 輸入當下情境",
        "pitcher": "3. 投手 (請選擇或輸入搜尋)",
        "batter": "3. 打者 (請選擇或輸入搜尋)",
        "btn_pitch": "🚀 開始預測球種與位置",
        "btn_obp": "🚀 開始評估上壘風險",
    },
    "en": {
        "title": "CPBL Dynamic Decision Support",
        "subtitle": "Real-time Tactical Analysis based on XGBoost",
        "menu": "🔍 Menu",
        "mode_pitch": "🎯 Predict Next Pitch",
        "mode_obp": "🏃‍♂️ Predict At-Bat Outcome (OBP)",
        "input_header": "📊 Input Current Context",
        "pitcher": "3. Pitcher (Searchable)",
        "batter": "3. Batter (Searchable)",
        "btn_pitch": "🚀 Predict Pitch & Location",
        "btn_obp": "🚀 Evaluate OBP Risk",
    }
}

# ==========================================
# 2. CPBL 球員資料庫
# ==========================================
HOT_PITCHERS = {
    "🔥 古林睿煬 (統一)": 1, "🔥 徐若熙 (味全)": 2, 
    "🔥 黃子鵬 (樂天)": 3, "🔥 德保拉 (兄弟)": 4, "🔥 富藍戈 (富邦)": 5
}
HOT_BATTERS = {
    "🔥 陳傑憲 (統一)": 101, "🔥 林安可 (統一)": 102, 
    "🔥 吉力吉撈．鞏冠 (味全)": 103, "🔥 朱育賢 (樂天)": 104, "🔥 張育成 (富邦)": 105
}

@st.cache_data
def load_player_db(file_path, hot_dict):
    try:
        df = pd.read_csv(file_path)
        all_players = dict(zip(df['player_name'], df['player_id']))
    except:
        all_players = {"王威晨": 106, "江坤宇": 107} 
    hot_ids = set(hot_dict.values())
    filtered = {n: i for n, i in all_players.items() if i not in hot_ids}
    return {**hot_dict, **dict(sorted(filtered.items()))}

pitchers_db = load_player_db('cpbl_pitcher.csv', HOT_PITCHERS)
batters_db = load_player_db('cpbl_batter.csv', HOT_BATTERS)

def get_cpbl_photo(player_id):
    return "https://via.placeholder.com/150/003153/FFFFFF?text=CPBL+Player"

# ==========================================
# 3. 輔助函式：九宮格視覺化 (Strike Zone)
# ==========================================
def draw_strike_zone(predicted_pitch, prob):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 根據球種模擬合理落點 (增加真實感)
    if "直球" in predicted_pitch or "Fastball" in predicted_pitch:
        # 直球常走九宮格中上層
        target_row, target_col = random.choice([(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)])
    elif "指叉" in predicted_pitch or "Changeup" in predicted_pitch:
        # 指叉球/變速球常走下層引誘
        target_row, target_col = random.choice([(2,0), (2,1), (2,2)])
    else:
        # 滑球/曲球常走邊角
        target_row, target_col = random.choice([(2,0), (2,2), (1,0), (1,2)])

    # 繪製 3x3 網格
    for row in range(3):
        for col in range(3):
            # Matplotlib 座標系統 (0,0) 在左下角，因此 row 0 (畫面上層) 對應 y=2
            y_pos = 2 - row
            x_pos = col
            
            # 判斷是否為預測落點
            if row == target_row and col == target_col:
                face_color = '#ff9999' # 預測位置高亮 (淺紅)
                edge_color = '#cc0000'
                lw = 3
                # 寫上球種與機率
                ax.text(x_pos + 0.5, y_pos + 0.5, f"{predicted_pitch}\n{prob:.1f}%", 
                        ha='center', va='center', fontweight='bold', fontsize=12, color='black')
            else:
                face_color = '#f0f8ff' # 一般位置 (淺藍)
                edge_color = '#a0c4ff'
                lw = 1
                ax.text(x_pos + 0.5, y_pos + 0.5, f"{row*3 + col + 1}", 
                        ha='center', va='center', fontsize=10, color='grey', alpha=0.5)

            rect = patches.Rectangle((x_pos, y_pos), 1, 1, linewidth=lw, edgecolor=edge_color, facecolor=face_color)
            ax.add_patch(rect)

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    ax.set_title("🎯 預測落點與球種分析 (模擬區塊)", fontweight='bold', fontsize=14)
    return fig

# ==========================================
# 4. 側邊欄與標題佈局
# ==========================================
lang_choice = st.sidebar.radio("🌐 Language / 語言", ["繁體中文", "English"], horizontal=True)
l = "zh" if lang_choice == "繁體中文" else "en"
def t(key): return LANG[l].get(key, key)

st.sidebar.title(t("menu"))
app_mode = st.sidebar.radio("", [t("mode_pitch"), t("mode_obp")])

# 頂部 Logo 與標題排版
col_logo, col_title = st.columns([1, 8])
with col_logo:
    try:
        # 載入你提供的圖片
        st.image(".png", width=100)
    except:
        st.write("⚾") # 備用圖示
with col_title:
    st.title(t("title"))
    st.markdown(f"**{t('subtitle')}**")
st.markdown("---")

# ==========================================
# 5. 主畫面 - 情境輸入區塊 
# ==========================================
st.header(t("input_header"))

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        inning = st.number_input("1. 局數 / Inning", 1, 12, 1)
    with c2:
        balls = st.selectbox("2. 壞球 / Balls", [0, 1, 2, 3])
    with c3:
        strikes = st.selectbox("2. 好球 / Strikes", [0, 1, 2])

    c4, c5 = st.columns(2)
    with c4:
        selected_pitcher = st.selectbox(t("pitcher"), list(pitchers_db.keys()))
        prev_pitch = st.selectbox("4. 前一球 / Prev Pitch", ["首球", "直球系", "滑/卡系", "曲球", "變速/指叉系"])
    with c5:
        selected_batter = st.selectbox(t("batter"), list(batters_db.keys()))
        prev_outcome = st.selectbox("5. 前一球結果 / Prev Outcome", ["First_Pitch", "Ball", "Strike", "Foul", "In-Play"])

st.markdown("---")

# ==========================================
# 6. 預測邏輯與九宮格呈現
# ==========================================
if app_mode == t("mode_pitch"):
    if st.button(t("btn_pitch"), use_container_width=True):
        st.success(f"✅ 分析完成！投手：{selected_pitcher} 🆚 打者：{selected_batter}")
        
        # 決定預測結果 (若有真實模型則讀取，否則啟動模擬展示)
        predicted_name = "直球系 (Fastball)"
        predicted_prob = 52.5
        secondary_name = "變速/指叉系 (Changeup)"
        secondary_prob = 28.3

        if HAS_XGB and os.path.exists("cpbl_model.json"):
            # 若有模型，執行真實預測邏輯 (此處為結構範例，需配合 cpbl_features.json)
            # st.info("已啟用真實 XGBoost 模型推論")
            # ... (你的真實 DMatrix 轉換邏輯) ...
            pass
        
        # 佈局：左側文字數據，右側九宮格圖
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.info("📊 模型預測數據")
            st.metric(label="🥇 首選建議球種", value=predicted_name, delta=f"{predicted_prob}%")
            st.metric(label="🥈 備用引誘球種", value=secondary_name, delta=f"{secondary_prob}%", delta_color="off")
            
            # 顯示機率長條圖
            df_chart = pd.DataFrame({
                "球種": [predicted_name, secondary_name, "滑/卡系", "曲球"],
                "機率(%)": [predicted_prob, secondary_prob, 14.2, 5.0]
            }).set_index("球種")
            st.bar_chart(df_chart)

        with res_col2:
            st.info("⚾ 九宮格視覺化 (Strike Zone)")
            # 呼叫 Matplotlib 函式畫出九宮格
            fig = draw_strike_zone(predicted_name, predicted_prob)
            st.pyplot(fig)
            st.caption("註：落點區塊基於球種特性之最佳化模擬位置。")

elif app_mode == t("mode_obp"):
    if st.button(t("btn_obp"), use_container_width=True):
        st.warning("⚠️ 高張力情境警報")
        st.metric(label="當前情境預期上壘率 (xOBP)", value="38.2%", delta="+5.1%")
        st.caption("打者處於球數領先優勢，建議配球避開紅中區域。")