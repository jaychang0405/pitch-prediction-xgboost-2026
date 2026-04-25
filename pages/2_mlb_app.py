# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="MLB Dynamic Prediction", page_icon="⚾", layout="wide")

# ==========================================
# 0. 簡易密碼鎖
# ==========================================
pwd = st.text_input("🔒 請輸入存取密碼", type="password")
if pwd != "20050405":  # 這裡換成你想要的密碼
    st.warning("請輸入正確密碼以解鎖系統。")
    st.stop()  # 密碼錯誤時，程式會在這裡停止，完全不會顯示下面的內容

# ==========================================
# 1. 雙語字典設定
# ==========================================
LANG = {
    "zh": {
        "title": "⚾ 職棒動態決策支援系統",
        "subtitle": "基於 XGBoost 與投球序列特徵的即時戰術分析",
        "menu": "🔍 功能選單",
        "mode_pitch": "🎯 預測下一球球種",
        "mode_obp": "🏃‍♂️ 預測打擊結果 (上壘率)",
        "input_header": "📊 輸入當下情境",
        "pitcher": "3. 投手 (輸入關鍵字搜尋，支援捲動)",
        "batter": "3. 打者 (輸入關鍵字搜尋，支援捲動)",
        "btn_pitch": "🚀 開始預測球種",
        "btn_obp": "🚀 開始評估上壘風險",
    },
    "en": {
        "title": "⚾ MLB Dynamic Decision Support System",
        "subtitle": "Real-time Tactical Analysis based on XGBoost & Pitch Sequencing",
        "menu": "🔍 Menu",
        "mode_pitch": "🎯 Predict Next Pitch",
        "mode_obp": "🏃‍♂️ Predict At-Bat Outcome (OBP)",
        "input_header": "📊 Input Current Context",
        "pitcher": "3. Pitcher (Type to search, scrollable)",
        "batter": "3. Batter (Type to search, scrollable)",
        "btn_pitch": "🚀 Predict Pitch",
        "btn_obp": "🚀 Evaluate OBP Risk",
    }
}

# ==========================================
# 2. 球員資料庫 (Name -> MLB ID) 從 CSV 讀取
# ==========================================
import os

# --- 投手名單 ---
# 1. 保留熱門名單 (為了讓他們強制排在最前面，並加上 🔥)
HOT_PITCHERS = {
    "🔥 Shohei Ohtani (Pitcher)": 660271,
    "🔥 Gerrit Cole": 543037,
    "🔥 Yoshinobu Yamamoto": 808967,
    "🔥 Justin Verlander": 434378,
    "🔥 Corbin Burnes": 669203
}

# 2. 讀取 CSV 檔案 (動態抓取全聯盟投手)
try:
    # 假設你的 CSV 欄位名稱是 'player_name' 和 'pitcher' (ID)
    df_pitchers = pd.read_csv('pitcher_dict.csv')
    # 將 DataFrame 轉成字典：{'Aaron Nola': 605400, ...}
    ALL_PITCHERS = dict(zip(df_pitchers['player_name'], df_pitchers['pitcher']))
except FileNotFoundError:
    # 萬一檔案還沒放進來，先給個預設空字典，才不會報錯
    ALL_PITCHERS = {"Aaron Nola": 605400, "Blake Snell": 605483} 

# 3. 合併字典並去重
# 找出熱門名單中已經有的 ID，避免選單出現兩個 Gerrit Cole
hot_pitcher_ids = set(HOT_PITCHERS.values())
filtered_all_pitchers = {name: pid for name, pid in ALL_PITCHERS.items() if pid not in hot_pitcher_ids}

# 合併：熱門排前面，剩下的按照字母排序
pitchers_db = {**HOT_PITCHERS, **dict(sorted(filtered_all_pitchers.items()))}
pitcher_names = list(pitchers_db.keys())


# --- 打者名單 ---
HOT_BATTERS = {
    "🔥 Mike Trout": 545361,
    "🔥 Rafael Devers": 646240,
    "🔥 Aaron Judge": 592450,
    "🔥 Shohei Ohtani (Batter)": 660271,
    "🔥 Juan Soto": 665742
}

try:
    # 假設你也有輸出打者的 CSV
    df_batters = pd.read_csv('batter_dict.csv')
    ALL_BATTERS = dict(zip(df_batters['player_name'], df_batters['batter']))
except FileNotFoundError:
    ALL_BATTERS = {"Austin Riley": 663624, "Bo Bichette": 666182}

hot_batter_ids = set(HOT_BATTERS.values())
filtered_all_batters = {name: pid for name, pid in ALL_BATTERS.items() if pid not in hot_batter_ids}

batters_db = {**HOT_BATTERS, **dict(sorted(filtered_all_batters.items()))}
batter_names = list(batters_db.keys())

# 獲取頭像網址的輔助函數
def get_headshot_url(player_id):
    return f"https://midfield.mlbstatic.com/v1/people/{player_id}/spots/120"

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
st.sidebar.markdown("---")

# ==========================================
# 4. 主畫面 - 情境輸入區塊 
# ==========================================
st.header(t("input_header"))

col_inputs, col_avatars = st.columns([2, 1])

with col_inputs:
    c1, c2, c3 = st.columns(3)
    with c1:
        inning = st.number_input("1. 局數 / Inning", min_value=1, max_value=12, value=1)
    with c2:
        balls = st.selectbox("2. 壞球 / Balls", [0, 1, 2, 3])
    with c3:
        strikes = st.selectbox("2. 好球 / Strikes", [0, 1, 2])

    c4, c5 = st.columns(2)
    with c4:
        # 選單包含長名單，可以直接捲動或打字搜尋
        selected_pitcher = st.selectbox(t("pitcher"), pitcher_names)
    with c5:
        selected_batter = st.selectbox(t("batter"), batter_names)

    c6, c7 = st.columns(2)
    with c6:
        prev_pitch = st.selectbox("4. 前一球 / Prev Pitch", ["First_Pitch", "Fastball_System", "Slider_Cutter", "Curveball", "Changeup"])
    with c7:
        prev_outcome = st.selectbox("5. 前一球結果 / Prev Outcome", ["First_Pitch", "ball", "called_strike", "swinging_strike", "foul", "hit_into_play"])

# 動態抓取 ID 並顯示大頭照
pitcher_id = pitchers_db[selected_pitcher]
batter_id = batters_db[selected_batter]

with col_avatars:
    st.markdown("<br>", unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        # 使用抓取到的 ID 組合圖片網址
        st.image(get_headshot_url(pitcher_id), caption="Pitcher", width=120)
    with a2:
        st.image(get_headshot_url(batter_id), caption="Batter", width=120)

st.markdown("---")

# ==========================================
# 5. 預測結果展示 (模擬)
# ==========================================
if app_mode == t("mode_pitch"):
    st.subheader(t("mode_pitch"))
    if st.button(t("btn_pitch"), use_container_width=True):
        st.success("✅ 分析完成！")
        # 這裡放預測長條圖 (略，與上一版相同)
        st.write(f"預測 {selected_pitcher} 下一球對決 {selected_batter} 將投出 Fastball_System (52%)")

elif app_mode == t("mode_obp"):
    st.subheader(t("mode_obp"))
    if st.button(t("btn_obp"), use_container_width=True):
        st.warning("⚠️ 高張力情境！")
        # 這裡放上壘率指標 (略，與上一版相同)
        st.metric(label="預測上壘率", value="34.5%", delta="+3.2%")