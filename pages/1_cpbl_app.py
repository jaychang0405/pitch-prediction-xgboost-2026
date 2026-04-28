# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_PATH = "data_cpbl"  # 設定為獨立資料夾

st.set_page_config(
    page_title="CPBL 動態決策支援系統",
    page_icon=os.path.join(DATA_PATH, "tennis-ball.svg"),
    layout="wide"
)

# ==========================================
# 0. 密碼鎖與路徑設定
# ==========================================
def clean_name(name):
    """輔助函式：去除掉下拉選單中為了美觀加上的火焰符號"""
    if isinstance(name, str):
        return name.replace("🔥 ", "").strip()
    return name

# ==========================================
# 1. 雙語字典設定 
# ==========================================
LANG = {
    "zh": {
        "title": "中職動態決策支援系統",
        "subtitle": "基於 XGBoost 的球種預測與上壘率 (OBP) 分析",
        "menu": "功能選單",
        "mode_pitch": " 預測下一球球種",
        "mode_obp": " 預測打擊結果 (上壘率)",
    },
    "en": {
        "title": "CPBL Dynamic Decision Support",
        "subtitle": "Pitch Prediction & OBP Analysis based on XGBoost",
        "menu": "Menu",
        "mode_pitch": " Predict Next Pitch",
        "mode_obp": " Predict At-Bat Outcome (OBP)",
    }
}

# ==========================================
# 2. 強化版資料與模型載入函式
# ==========================================
@st.cache_data
def load_cpbl_data():
    data_dict = {"pitch": {}, "obp_list": {}, "obp_db_dict": {}, "features": []}
    
    try:
        # --- 球種預測用 ---
        data_dict["pitch"]["p"] = pd.read_csv(os.path.join(DATA_PATH, 'cpbl_pitcher.csv'))
        data_dict["pitch"]["b"] = pd.read_csv(os.path.join(DATA_PATH, 'cpbl_batter.csv'))
        
        with open(os.path.join(DATA_PATH, 'cpbl_features.json'), 'r', encoding='utf-8') as f:
            data_dict["features"] = json.load(f)

        # --- OBP 預測用 ---
        # 1. 名單檔案 (選單顯示用)
        data_dict["obp_list"]["p"] = pd.read_csv(os.path.join(DATA_PATH, 'cpbl_pitcher_obp.csv'))
        data_dict["obp_list"]["b"] = pd.read_csv(os.path.join(DATA_PATH, 'cpbl_batter_obp.csv'))
        
        # 2. 數據庫檔案 (轉換為 Dictionary 方便 OBP 模型快速查表)
        df_p_obp_db = pd.read_csv(os.path.join(DATA_PATH, 'cpbl_pitcher_obp_db.csv'))
        df_b_obp_db = pd.read_csv(os.path.join(DATA_PATH, 'cpbl_batter_obp_db.csv'))
        
        data_dict["obp_db_dict"]["p"] = dict(zip(df_p_obp_db['player_name'].astype(str), df_p_obp_db['avg_obp_allowed']))
        data_dict["obp_db_dict"]["b"] = dict(zip(df_b_obp_db['player_name'].astype(str), df_b_obp_db['avg_obp']))
        
    except Exception as e:
        st.error(f"⚠️ 資料載入發生錯誤，請檢查 {DATA_PATH} 資料夾內的檔案。錯誤訊息: {e}")
        
    return data_dict

@st.cache_resource
def load_models():
    pitch_model, obp_model = None, None
    if HAS_XGB:
        if os.path.exists(os.path.join(DATA_PATH, "cpbl_pitch_model.json")):
            pitch_model = xgb.Booster()
            pitch_model.load_model(os.path.join(DATA_PATH, "cpbl_pitch_model.json"))
        if os.path.exists(os.path.join(DATA_PATH, "cpbl_obp_model.json")):
            obp_model = xgb.Booster()
            obp_model.load_model(os.path.join(DATA_PATH, "cpbl_obp_model.json"))
    return pitch_model, obp_model

data = load_cpbl_data()
pitch_model, obp_model = load_models()

# ==========================================
# 3. 輔助函式：九宮格視覺化 (球種用)
# ==========================================
def draw_strike_zone(predicted_pitch_en, prob):
    fig, ax = plt.subplots(figsize=(5, 5))
    if "Fastball" in predicted_pitch_en:
        target_row, target_col = random.choice([(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)])
    elif "Changeup" in predicted_pitch_en or "Splitter" in predicted_pitch_en:
        target_row, target_col = random.choice([(2,0), (2,1), (2,2)])
    else:
        target_row, target_col = random.choice([(2,0), (2,2), (1,0), (1,2)])

    for row in range(3):
        for col in range(3):
            y_pos, x_pos = 2 - row, col
            if row == target_row and col == target_col:
                face_color, edge_color, lw = '#ff9999', '#cc0000', 3
                ax.text(x_pos + 0.5, y_pos + 0.5, f"{predicted_pitch_en}\n{prob:.1f}%", ha='center', va='center', fontweight='bold', fontsize=12, color='black')
            else:
                face_color, edge_color, lw = '#f0f8ff', '#a0c4ff', 1
                ax.text(x_pos + 0.5, y_pos + 0.5, f"{row*3 + col + 1}", ha='center', va='center', fontsize=10, color='grey', alpha=0.5)

            rect = patches.Rectangle((x_pos, y_pos), 1, 1, linewidth=lw, edgecolor=edge_color, facecolor=face_color)
            ax.add_patch(rect)

    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5); ax.axis('off')
    ax.set_title("Predicted Pitch & Location", fontweight='bold', fontsize=14)
    return fig

# ==========================================
# 4. 頁面標題列 (含右上角語言切換)
# ==========================================
# 使用 columns 將畫面切分為 Logo、標題、語言選擇器
# 比例分配：1 (Logo) : 6 (標題) : 2 (語言)
col_logo, col_title, col_lang = st.columns([1, 6, 2])

with col_lang:
    # 語言選擇移至此處，使用 label_visibility="collapsed" 隱藏標籤
    lang_choice = st.radio(
        "🌐 Language", 
        ["繁體中文", "English"], 
        horizontal=True, 
        label_visibility="collapsed"
    )
    l = "zh" if lang_choice == "繁體中文" else "en"
    def t(key): return LANG[l].get(key, key)

with col_logo:
    try:
        st.image(os.path.join(DATA_PATH, "cpbl_icon.png"), width=100)
    except:
        st.write("⚾")

with col_title:
    st.title(t("title"))
    st.markdown(f"**{t('subtitle')}**")

# 側邊欄僅保留功能選單
st.sidebar.title(t("menu"))
app_mode = st.sidebar.radio("", [t("mode_pitch"), t("mode_obp")])
st.sidebar.markdown("---")

# ==========================================
# 5. 共用情境輸入區塊 
# ==========================================
st.header(" 輸入當下比賽情境")

# 動態切換下拉選單名單
if app_mode == t("mode_pitch"):
    p_names = data["pitch"]["p"]["player_name"].tolist() if "p" in data["pitch"] else ["無資料"]
    b_names = data["pitch"]["b"]["player_name"].tolist() if "b" in data["pitch"] else ["無資料"]
else:
    p_names = data["obp_list"]["p"]["player_name"].tolist() if "p" in data["obp_list"] else ["無資料"]
    b_names = data["obp_list"]["b"]["player_name"].tolist() if "b" in data["obp_list"] else ["無資料"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    inning = st.number_input("1. 局數 / Inning", 1, 12, 1)
with c2:
    balls = st.selectbox("2. 壞球 / Balls", [0, 1, 2, 3])
with c3:
    strikes = st.selectbox("3. 好球 / Strikes", [0, 1, 2])
with c4:
    outs = st.selectbox("4. 出局數 / Outs", [0, 1, 2]) # 兩模式共用出局數

c5, c6 = st.columns(2)
with c5:
    selected_pitcher = st.selectbox("投手 (Pitcher)", p_names)
    clean_pitcher = clean_name(selected_pitcher)
with c6:
    selected_batter = st.selectbox("打者 (Batter)", b_names)
    clean_batter = clean_name(selected_batter)

# ==========================================
# 6. 預測邏輯 (Pitch & OBP)
# ==========================================

# ------------------------------------------
# 模式 A: 球種預測
# ------------------------------------------
if app_mode == t("mode_pitch"):
    st.markdown("#### 球種序列特徵")
    c7, c8 = st.columns(2)
    with c7: prev_pitch = st.selectbox("前一球球種", ["首球", "直球系", "滑/卡系", "曲球", "變速/指叉系"])
    with c8: prev_outcome = st.selectbox("前一球結果", ["First_Pitch", "Ball", "Strike", "Foul", "In-Play"])
    st.markdown("---")

    if st.button(" 開始預測球種與位置", use_container_width=True):
        st.success(f" 分析完成！投手：{clean_pitcher} vs 打者：{clean_batter}")
        
        # 預設數據
        ui_predicted_name, ui_secondary_name = "直球系 (Fastball)", "變速/指叉系 (Changeup)"
        plot_predicted_name_en, predicted_prob, secondary_prob = "Fastball", 52.5, 28.3
        chart_names = [ui_predicted_name, ui_secondary_name, "滑/卡系 (Slider/Cutter)", "曲球 (Curveball)"]
        chart_probs = [predicted_prob, secondary_prob, 14.2, 5.0]

        if HAS_XGB and pitch_model and data["features"]:
            try:
                PITCH_CLASSES = ['Changeup', 'Curveball', 'Fastball_System', 'Slider_Cutter']
                PITCH_TRANS_UI = {'Changeup': '變速/指叉系', 'Curveball': '曲球', 'Fastball_System': '直球系', 'Slider_Cutter': '滑/卡系'}
                expected_features = data["features"]
                
                input_df = pd.DataFrame(0.0, index=[0], columns=expected_features)
                if 'outs' in expected_features: input_df['outs'] = outs
                if 'pitch_count' in expected_features: input_df['pitch_count'] = 15
                
                count_col = f"count_{balls}-{strikes}"
                if count_col in expected_features: input_df[count_col] = 1.0
                    
                prev_pitch_map = {"首球": "First_Pitch", "直球系": "Fastball_System", "滑/卡系": "Slider_Cutter", "曲球": "Curveball", "變速/指叉系": "Changeup"}
                prev_pitch_col = f"prev_grouped_pitch_{prev_pitch_map.get(prev_pitch, 'First_Pitch')}"
                if prev_pitch_col in expected_features: input_df[prev_pitch_col] = 1.0
                    
                dtest = xgb.DMatrix(input_df.to_numpy(dtype='float32'))
                probs = pitch_model.predict(dtest)[0]
                
                top_indices = np.argsort(probs)[::-1]
                best_idx, second_idx = top_indices[0], top_indices[1]
                
                ui_predicted_name = PITCH_TRANS_UI[PITCH_CLASSES[best_idx]]
                ui_secondary_name = PITCH_TRANS_UI[PITCH_CLASSES[second_idx]]
                plot_predicted_name_en = PITCH_CLASSES[best_idx].split('_')[0]
                predicted_prob = float(probs[best_idx] * 100)
                secondary_prob = float(probs[second_idx] * 100)
                chart_probs = [float(p * 100) for p in probs]
                chart_names = [PITCH_TRANS_UI[c] for c in PITCH_CLASSES]

            except Exception as e:
                st.warning(f"⚠️ 預測發生錯誤，顯示模擬數據。錯誤: {e}")

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.info("模型預測數據")
            st.metric(label="1. 首選建議球種", value=ui_predicted_name, delta=f"{predicted_prob:.1f}%")
            st.metric(label="2. 備用引誘球種", value=ui_secondary_name, delta=f"{secondary_prob:.1f}%", delta_color="off")
            df_chart = pd.DataFrame({"球種": chart_names, "機率(%)": chart_probs}).set_index("球種")
            st.bar_chart(df_chart)
        with res_col2:
            st.info("九宮格視覺化 (Strike Zone)")
            fig = draw_strike_zone(plot_predicted_name_en, predicted_prob)
            st.pyplot(fig)

# ------------------------------------------
# 模式 B: 上壘率預測
# ------------------------------------------
elif app_mode == t("mode_obp"):
    st.markdown("#### 模型進階特徵 (XGBoost 必備欄位)")
    
    r1, r2, r3 = st.columns(3)
    with r1:
        score_diff = st.number_input("比分差 (主隊減客隊，領先為正)", value=0)
    with r2:
        base_states = {
            "無人在壘": (0, 0), "一壘有人": (1, 1), "二壘有人": (1, 2), "一二壘有人": (2, 3),
            "三壘有人": (1, 4), "一三壘有人": (2, 5), "二三壘有人": (2, 6), "滿壘": (3, 7)
        }
        selected_base = st.selectbox("壘上局面", list(base_states.keys()))
        runners_on_base, base_state_code = base_states[selected_base]
    with r3:
        pitch_count = st.number_input("投手當前用球數", 0, 150, 15)

    r4, r5 = st.columns(2)
    with r4:
        is_home = st.toggle("主場球隊打擊 (Is Home Team)", value=True)
    with r5:
        platoon = st.toggle("投打對決優勢 (Platoon Adv.)", value=False, help="若右打對左投，或左打對右投，請開啟此項")
    
    st.markdown("---")

    if st.button(" 開始評估上壘風險", use_container_width=True):
        if obp_model is None:
            st.error("找不到模型檔案，請確認 `cpbl_obp_model.json` 已放置於 data_cpbl 資料夾中。")
        else:
            # 查表：若遇新秀或查無資料則代入 CPBL 預設平均上壘率 0.330
            hist_b_obp = data["obp_db_dict"]["b"].get(clean_batter, 0.330)
            hist_p_obp = data["obp_db_dict"]["p"].get(clean_pitcher, 0.330)
            
            st.info(f"系統自動偵測帶入：{clean_batter} 歷史 OBP ({hist_b_obp:.3f}) / {clean_pitcher} 歷史被 OBP ({hist_p_obp:.3f})")

            # 嚴格對齊組員原本的 12 項特徵順序與欄位名稱
            feature_names = ['balls', 'strikes', 'outs_when_up', 'inning', 'score_diff', 'runners_on_base', 
                             'pitch_count', 'batter_hist_obp', 'pitcher_hist_obp_allowed', 'is_home_team', 
                             'platoon_advantage', 'base_state_code']
            
            feature_values = [
                balls, strikes, outs, inning, score_diff, runners_on_base,
                pitch_count, hist_b_obp, hist_p_obp, 1 if is_home else 0, 1 if platoon else 0, base_state_code
            ]
            
            try:
                # 轉換為 DMatrix 並進行精準預測
                df_input = pd.DataFrame([feature_values], columns=feature_names)
                dmatrix = xgb.DMatrix(df_input)
                prob = obp_model.predict(dmatrix)[0]
                
                # 顯示結果
                st.metric(label=f"預測 {clean_batter} 該打席上壘機率 (xOBP)", value=f"{prob:.1%}")
                
                if prob > 0.35:
                    st.warning("⚠️ 高上壘風險！打者具備優勢，建議投手採取邊角引誘球策略。")
                else:
                    st.success("✅ 目前對戰情境對投手有利。")
            except Exception as e:
                st.error(f"⚠️ OBP 模型推論失敗，請檢查資料格式。錯誤: {e}")