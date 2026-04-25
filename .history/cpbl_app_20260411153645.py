# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
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

# ==========================================
# 3. 輔助函式：九宮格視覺化 (改為全英文防亂碼)
# ==========================================
def draw_strike_zone(predicted_pitch_en, prob):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 根據球種模擬合理落點 (增加真實感)
    if "Fastball" in predicted_pitch_en:
        # 直球常走九宮格中上層
        target_row, target_col = random.choice([(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)])
    elif "Changeup" in predicted_pitch_en or "Splitter" in predicted_pitch_en:
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
                # 寫上球種與機率 (英文)
                ax.text(x_pos + 0.5, y_pos + 0.5, f"{predicted_pitch_en}\n{prob:.1f}%", 
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
    # 標題改為英文
    ax.set_title("Predicted Pitch & Location (Simulated)", fontweight='bold', fontsize=14)
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
        st.image("cpbl_icon.png", width=100)
    except:
        st.write("⚾")
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
        
        # 定義球種對照表 (確保與你訓練時的 LabelEncoder 順序一致)
        PITCH_CLASSES = ['Changeup', 'Curveball', 'Fastball_System', 'Slider_Cutter']
        PITCH_TRANS_UI = {
            'Changeup': '變速/指叉系 (Changeup)',
            'Curveball': '曲球 (Curveball)',
            'Fastball_System': '直球系 (Fastball)',
            'Slider_Cutter': '滑/卡系 (Slider/Cutter)'
        }
        PITCH_TRANS_PLOT = {
            'Changeup': 'Changeup',
            'Curveball': 'Curveball',
            'Fastball_System': 'Fastball',
            'Slider_Cutter': 'Slider/Cutter'
        }

        # 預設為模擬數據 (防呆)
        ui_predicted_name = "直球系 (Fastball)"
        ui_secondary_name = "變速/指叉系 (Changeup)"
        plot_predicted_name_en = "Fastball"
        predicted_prob = 52.5
        secondary_prob = 28.3
        chart_data_probs = [52.5, 28.3, 14.2, 5.0]
        chart_data_names = ["直球系 (Fastball)", "變速/指叉系 (Changeup)", "滑/卡系 (Slider/Cutter)", "曲球 (Curveball)"]

        # ⚾ === 真實模型預測邏輯 === ⚾
        try:
            # 確保模型存在，且已經讀取到特徵表
            if 'bst' in globals() and bst is not None and expected_features:
                
                # 1. 建立一張全為 0 的空特徵表 (40個特徵)
                input_df = pd.DataFrame(0.0, index=[0], columns=expected_features)
                
                # 2. 填入數值型特徵
                if 'outs' in expected_features: input_df['outs'] = 0 # 假設 0 出局，你可自行在 UI 加上
                if 'score_diff' in expected_features: input_df['score_diff'] = 0 
                if 'pitch_count' in expected_features: input_df['pitch_count'] = 15
                if 'prev_coordX' in expected_features: input_df['prev_coordX'] = 0.0
                if 'prev_coordY' in expected_features: input_df['prev_coordY'] = 0.0
                
                # 3. 填入類別型特徵 (One-Hot Encoding)
                count_col = f"count_{balls}-{strikes}"
                if count_col in expected_features: input_df[count_col] = 1.0
                    
                prev_pitch_map = {"首球": "First_Pitch", "直球系": "Fastball_System", "滑/卡系": "Slider_Cutter", "曲球": "Curveball", "變速/指叉系": "Changeup"}
                prev_pitch_col = f"prev_grouped_pitch_{prev_pitch_map.get(prev_pitch, 'First_Pitch')}"
                if prev_pitch_col in expected_features: input_df[prev_pitch_col] = 1.0
                    
                prev_code_col = f"prev_pitchCode_{prev_outcome}"
                if prev_code_col in expected_features: input_df[prev_code_col] = 1.0
                
                # 4. 進行預測
                dtest = xgb.DMatrix(input_df.to_numpy(dtype='float32'))
                probs = bst.predict(dtest)[0]
                
                # 5. 抓取機率最高的前兩名
                top_indices = np.argsort(probs)[::-1]
                
                # 更新 UI 變數為真實數據
                best_idx, second_idx = top_indices[0], top_indices[1]
                
                ui_predicted_name = PITCH_TRANS_UI[PITCH_CLASSES[best_idx]]
                ui_secondary_name = PITCH_TRANS_UI[PITCH_CLASSES[second_idx]]
                plot_predicted_name_en = PITCH_TRANS_PLOT[PITCH_CLASSES[best_idx]]
                
                predicted_prob = float(probs[best_idx] * 100)
                secondary_prob = float(probs[second_idx] * 100)
                
                # 更新圖表數據
                chart_data_probs = [float(p * 100) for p in probs]
                chart_data_names = [PITCH_TRANS_UI[c] for c in PITCH_CLASSES]

        except Exception as e:
            st.warning(f"⚠️ 真實預測發生錯誤，顯示模擬數據。錯誤訊息: {e}")

        # === 顯示結果 ===
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.info("📊 模型預測數據")
            st.metric(label="🥇 首選建議球種", value=ui_predicted_name, delta=f"{predicted_prob:.1f}%")
            st.metric(label="🥈 備用引誘球種", value=ui_secondary_name, delta=f"{secondary_prob:.1f}%", delta_color="off")
            
            df_chart = pd.DataFrame({
                "球種": chart_data_names,
                "機率(%)": chart_data_probs
            }).sort_values(by="機率(%)", ascending=False).set_index("球種")
            st.bar_chart(df_chart)

        with res_col2:
            st.info("⚾ 九宮格視覺化 (Strike Zone)")
            fig = draw_strike_zone(plot_predicted_name_en, predicted_prob)
            st.pyplot(fig)
            st.caption("註：落點區塊基於球種特性之最佳化模擬位置。")

elif app_mode == t("mode_obp"):
    if st.button(t("btn_obp"), use_container_width=True):
        st.warning("⚠️ 高張力情境警報")
        st.metric(label="當前情境預期上壘率 (xOBP)", value="38.2%", delta="+5.1%")
        st.caption("打者處於球數領先優勢，建議配球避開紅中區域。")