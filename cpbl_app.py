# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="CPBL Dynamic Prediction", page_icon="⚾", layout="wide")

# ==========================================
# 1. 雙語字典設定 (調整為 CPBL 用語)
# ==========================================
LANG = {
    "zh": {
        "title": "⚾ CPBL 中職動態決策支援系統",
        "subtitle": "基於 XGBoost 與投球序列特徵的即時戰術分析",
        "menu": "🔍 功能選單",
        "mode_pitch": "🎯 預測下一球球種",
        "mode_obp": "🏃‍♂️ 預測打擊結果 (上壘率)",
        "input_header": "📊 輸入當下情境",
        "pitcher": "3. 投手 (請選擇或輸入搜尋)",
        "batter": "3. 打者 (請選擇或輸入搜尋)",
        "btn_pitch": "🚀 開始預測球種",
        "btn_obp": "🚀 開始評估上壘風險",
    },
    "en": {
        "title": "⚾ CPBL Dynamic Decision Support System",
        "subtitle": "Real-time Tactical Analysis based on XGBoost & Pitch Sequencing",
        "menu": "🔍 Menu",
        "mode_pitch": "🎯 Predict Next Pitch",
        "mode_obp": "🏃‍♂️ Predict At-Bat Outcome (OBP)",
        "input_header": "📊 Input Current Context",
        "pitcher": "3. Pitcher (Searchable)",
        "batter": "3. Batter (Searchable)",
        "btn_pitch": "🚀 Predict Pitch",
        "btn_obp": "🚀 Evaluate OBP Risk",
    }
}

# ==========================================
# 2. CPBL 球員資料庫 (建議從 CSV 讀取)
# ==========================================
# 註：CPBL 無官方開放 API 頭像，此處 ID 供模型辨識使用
HOT_PITCHERS = {
    "🔥 古林睿煬 (統一)": 1,
    "🔥 徐若熙 (味全)": 2,
    "🔥 黃子鵬 (樂天)": 3,
    "🔥 德保拉 (兄弟)": 4,
    "🔥 富藍戈 (富邦)": 5
}

HOT_BATTERS = {
    "🔥 陳傑憲 (統一)": 101,
    "🔥 林安可 (統一)": 102,
    "🔥 吉力吉撈．鞏冠 (味全)": 103,
    "🔥 朱育賢 (樂天)": 104,
    "🔥 張育成 (富邦)": 105
}

# 讀取 CPBL 字典檔案 (需自行準備 cpbl_pitcher.csv 和 cpbl_batter.csv)
@st.cache_data
def load_player_db(file_path, hot_dict):
    try:
        df = pd.read_csv(file_path)
        all_players = dict(zip(df['player_name'], df['player_id']))
    except:
        all_players = {"王威晨": 106, "江坤宇": 107} # 備援資料
    
    # 過濾並排序
    hot_ids = set(hot_dict.values())
    filtered = {n: i for n, i in all_players.items() if i not in hot_ids}
    return {**hot_dict, **dict(sorted(filtered.items()))}

pitchers_db = load_player_db('cpbl_pitcher.csv', HOT_PITCHERS)
batters_db = load_player_db('cpbl_batter.csv', HOT_BATTERS)

# CPBL 暫時用佔位圖 (或你自己準備的 URL)
def get_cpbl_photo(player_id):
    # 如果你有球員照片資料庫，可改為：f"https://your-site.com/players/{player_id}.png"
    return "https://via.placeholder.com/150/003153/FFFFFF?text=CPBL+Player"

# ==========================================
# 3. 側邊欄與語言切換
# ==========================================
lang_choice = st.sidebar.radio("🌐 Language / 語言", ["繁體中文", "English"], horizontal=True)
l = "zh" if lang_choice == "繁體中文" else "en"
def t(key): return LANG[l].get(key, key)

st.title(t("title"))
st.markdown(f"**{t('subtitle')}**")

st.sidebar.title(t("menu"))
app_mode = st.sidebar.radio("", [t("mode_pitch"), t("mode_obp")])

# ==========================================
# 4. 主畫面 - 情境輸入區塊 
# ==========================================
st.header(t("input_header"))

col_inputs, col_avatars = st.columns([2, 1])

with col_inputs:
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
    with c5:
        selected_batter = st.selectbox(t("batter"), list(batters_db.keys()))

    c6, c7 = st.columns(2)
    with c6:
        # 針對 CPBL 常見球種微調
        prev_pitch = st.selectbox("4. 前一球 / Prev Pitch", ["首球", "直球", "滑球", "曲球", "變速球", "指叉球"])
    with c7:
        prev_outcome = st.selectbox("5. 前一球結果 / Prev Outcome", ["First_Pitch", "Ball", "Strike", "Foul", "In-Play"])

with col_avatars:
    st.write("") # 間距
    a1, a2 = st.columns(2)
    with a1:
        st.image(get_cpbl_photo(pitchers_db[selected_pitcher]), caption="Pitcher", width=120)
    with a2:
        st.image(get_cpbl_photo(batters_db[selected_batter]), caption="Batter", width=120)

st.markdown("---")

# ==========================================
# 5. 模擬預測邏輯
# ==========================================
if app_mode == t("mode_pitch"):
    if st.button(t("btn_pitch"), use_container_width=True):
        st.success("✅ 分析完成！")
        # 這裡模擬 CPBL 常見的配球數據
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.info(f"建議對決 {selected_batter} 的球種")
            # 這裡可以畫圖或寫結果
            st.write(f"1. **直球**: 45%")
            st.write(f"2. **指叉球**: 30%")
        with res_col2:
            st.info("熱區分析")
            st.write("建議位置：內角低處")

elif app_mode == t("mode_obp"):
    if st.button(t("btn_obp"), use_container_width=True):
        st.warning("⚠️ 高機率安打警報")
        st.metric(label="預期上壘率 (xOBP)", value="38.2%", delta="+5.1%")