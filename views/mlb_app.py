# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import ui_kit

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ==========================================
# 0. 檔案路徑設定
# ==========================================
DATA_PATH = "data_mlb"

# ==========================================
# 1. 雙語字典設定 (保持不變)
# ==========================================
LANG = {
    "zh": {
        "title": "MLB 動態決策支援系統",
        "subtitle": "基於 Statcast 與 XGBoost 的即時戰術分析",
        "menu": "功能選單",
        "mode_pitch": "🎯 預測下一球球種",
        "mode_obp": "🏃 預測打擊結果 (上壘率)",
        "input_header": "🎮 輸入當下比賽情境",
        "pitcher": "3. 投手 (Pitcher)",
        "batter": "3. 打者 (Batter)",
        "btn_pitch": "🎯 開始預測球種",
        "btn_obp": "🏃 評估上壘風險 (OBP)",
    },
    "en": {
        "title": "MLB Dynamic Decision Support",
        "subtitle": "Real-time Tactical Analysis based on Statcast & XGBoost",
        "menu": "Menu",
        "mode_pitch": "🎯 Predict Next Pitch",
        "mode_obp": "🏃 Predict At-Bat Outcome (OBP)",
        "input_header": "🎮 Input Current Context",
        "pitcher": "3. Pitcher",
        "batter": "3. Batter",
        "btn_pitch": "🎯 Predict Pitch",
        "btn_obp": "🏃 Evaluate OBP Risk",
    }
}

# ==========================================
# 2. 輔助函式 (讀取、載入) 保持不變
# ==========================================
def safe_read_csv(filename):
    path = os.path.join(DATA_PATH, filename)
    for enc in ['utf-8', 'utf-8-sig', 'big5', 'cp950', 'ansi']:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    return pd.DataFrame()

def safe_read_json(filename):
    path = os.path.join(DATA_PATH, filename)
    for enc in ['utf-8', 'utf-8-sig', 'big5', 'cp950']:
        try:
            with open(path, 'r', encoding=enc) as f: return json.load(f)
        except: continue
    return []

@st.cache_data
def load_mlb_dicts():
    HOT_PITCHERS = {"🔥 Shohei Ohtani (Pitcher)": 660271, "🔥 Gerrit Cole": 543037, "🔥 Yoshinobu Yamamoto": 808967, "🔥 Justin Verlander": 434378, "🔥 Corbin Burnes": 669203}
    df_pitchers = safe_read_csv('pitcher_dict.csv')
    ALL_PITCHERS = dict(zip(df_pitchers['player_name'], df_pitchers['pitcher'])) if not df_pitchers.empty else {}
    pitchers_db = {**HOT_PITCHERS, **dict(sorted({n: p for n, p in ALL_PITCHERS.items() if p not in set(HOT_PITCHERS.values())}.items()))}

    HOT_BATTERS = {"🔥 Mike Trout": 545361, "🔥 Rafael Devers": 646240, "🔥 Aaron Judge": 592450, "🔥 Shohei Ohtani (Batter)": 660271, "🔥 Juan Soto": 665742}
    df_batters = safe_read_csv('batter_dict.csv')
    ALL_BATTERS = dict(zip(df_batters['player_name'], df_batters['batter'])) if not df_batters.empty else {}
    batters_db = {**HOT_BATTERS, **dict(sorted({n: p for n, p in ALL_BATTERS.items() if p not in set(HOT_BATTERS.values())}.items()))}

    features = safe_read_json('mlb_pitch_features.json')
    pitch_classes = safe_read_json('mlb_pitch_classes.json')

    obp_db_dict = {"p": {}, "b": {}}
    df_p_obp = safe_read_csv('pitcher_stats_db.csv')
    df_b_obp = safe_read_csv('batter_stats_db.csv')

    if not df_p_obp.empty:
        cols = df_p_obp.columns
        obp_db_dict["p"] = dict(zip(df_p_obp[cols[0]].astype(str), df_p_obp[cols[1]]))
    if not df_b_obp.empty:
        cols = df_b_obp.columns
        obp_db_dict["b"] = dict(zip(df_b_obp[cols[0]].astype(str), df_b_obp[cols[1]]))

    return pitchers_db, batters_db, features, pitch_classes, obp_db_dict

@st.cache_resource
def load_mlb_models():
    p_model, o_model = None, None
    if HAS_XGB:
        if os.path.exists(os.path.join(DATA_PATH, "mlb_pitch_model.json")):
            p_model = xgb.Booster()
            p_model.load_model(os.path.join(DATA_PATH, "mlb_pitch_model.json"))
        if os.path.exists(os.path.join(DATA_PATH, "xgb_obp_model.json")):
            o_model = xgb.Booster()
            o_model.load_model(os.path.join(DATA_PATH, "xgb_obp_model.json"))
    return p_model, o_model

pitchers_db, batters_db, mlb_features, mlb_classes, obp_db_dict = load_mlb_dicts()
pitch_model, obp_model = load_mlb_models()

def get_headshot_url(player_id):
    return f"https://midfield.mlbstatic.com/v1/people/{player_id}/spots/120"

# ==========================================
# 3. 頁面標題列 (含右上角語言切換)
# ==========================================
col_title, col_lang = st.columns([6, 2])

with col_lang:
    lang_choice = st.radio(
        "🌐 Language",
        ["繁體中文", "English"],
        horizontal=True,
        label_visibility="collapsed"
    )
    l = "zh" if lang_choice == "繁體中文" else "en"
    def t(key): return LANG[l].get(key, key)

ui_kit.hero_banner(t("title"), t("subtitle"), icon="🇺🇸")

# ==========================================
# 4. 模式切換
# ==========================================
app_mode = st.segmented_control(
    t("menu"),
    [t("mode_pitch"), t("mode_obp")],
    default=t("mode_pitch"),
    label_visibility="collapsed",
)
if app_mode is None:
    app_mode = t("mode_pitch")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. 主畫面 - 情境輸入區塊 (邏輯與原本相同)
# ==========================================
st.header(t("input_header"))

with st.container(border=True):
    col_inputs, col_avatars = st.columns([2, 1])

    with col_inputs:
        c1, c2, c3, c4 = st.columns(4)
        with c1: inning = st.number_input("1. 局數 / Inning", 1, 12, 1)
        with c2: balls = st.selectbox("2. 壞球 / Balls", [0, 1, 2, 3])
        with c3: strikes = st.selectbox("3. 好球 / Strikes", [0, 1, 2])
        with c4: outs = st.selectbox("4. 出局數 / Outs", [0, 1, 2])

        c_base, c_pd1, c_pd2 = st.columns([2, 1, 1])
        with c_base:
            bases = st.multiselect("5. 壘上跑者", ["一壘 (1B)", "二壘 (2B)", "三壘 (3B)"])
            on_1b = 1 if "一壘 (1B)" in bases else 0
            on_2b = 1 if "二壘 (2B)" in bases else 0
            on_3b = 1 if "三壘 (3B)" in bases else 0
            runners_on_base = on_1b + on_2b + on_3b
            base_state_code = (on_1b * 1) + (on_2b * 2) + (on_3b * 4)

        with c_pd1:
            if app_mode == t("mode_pitch"): p_throws = st.selectbox("投手慣用手", ["R", "L"])
            else: score_diff = st.number_input("比分差", value=0)
        with c_pd2:
            if app_mode == t("mode_pitch"): stand = st.selectbox("打者站位", ["R", "L"])
            else: pitch_count = st.number_input("用球數", 1, 150, 15)

        c5, c6 = st.columns(2)
        with c5: selected_pitcher = st.selectbox(t("pitcher"), list(pitchers_db.keys()))
        with c6: selected_batter = st.selectbox(t("batter"), list(batters_db.keys()))

        if app_mode == t("mode_obp"):
            st.markdown("#### OBP 進階特徵 (XGBoost 必備)")
            o1, o2 = st.columns(2)
            with o1: is_home = st.toggle("主場球隊打擊 (Is Home Team)", value=True)
            with o2: platoon = st.toggle("投打對決優勢 (Platoon Adv.)", value=False)

        if app_mode == t("mode_pitch"):
            st.markdown("#### 球種序列特徵")
            c7, c8 = st.columns(2)
            with c7: prev_pitch = st.selectbox("前一球配球", ["First_Pitch", "Fastball_System", "Slider_Cutter", "Curveball", "Changeup"])
            with c8: prev_outcome = st.selectbox("前一球結果", ["First_Pitch", "ball", "called_strike", "swinging_strike", "foul", "hit_into_play", "other"])

    pitcher_id = pitchers_db.get(selected_pitcher, 660271)
    batter_id = batters_db.get(selected_batter, 660271)

    with col_avatars:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1: st.image(get_headshot_url(pitcher_id), caption="Pitcher", width=120)
        with a2: st.image(get_headshot_url(batter_id), caption="Batter", width=120)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. 真實 XGBoost 預測邏輯 (保持不變)
# ==========================================
if app_mode == t("mode_pitch"):
    if st.button(t("btn_pitch"), use_container_width=True, type="primary"):
        with st.spinner("AI 模型運算中..."):
            final_classes = mlb_classes if mlb_classes else ["Changeup", "Curveball", "Fastball_System", "Slider_Cutter"]
            final_probs = [10.0, 15.0, 50.0, 25.0]

            if HAS_XGB and pitch_model and mlb_features:
                try:
                    input_df = pd.DataFrame(0.0, index=[0], columns=mlb_features)
                    if 'inning' in mlb_features: input_df['inning'] = inning
                    if 'outs_when_up' in mlb_features: input_df['outs_when_up'] = outs
                    if 'pitch_number' in mlb_features: input_df['pitch_number'] = 15
                    if 'score_diff' in mlb_features: input_df['score_diff'] = 0
                    if 'on_1b' in mlb_features: input_df['on_1b'] = on_1b
                    if 'on_2b' in mlb_features: input_df['on_2b'] = on_2b
                    if 'on_3b' in mlb_features: input_df['on_3b'] = on_3b
                    if 'prev_plate_x' in mlb_features: input_df['prev_plate_x'] = 0.0
                    if 'prev_plate_z' in mlb_features: input_df['prev_plate_z'] = 2.5

                    count_col = f"count_{balls}-{strikes}"
                    if count_col in mlb_features: input_df[count_col] = 1.0
                    p_throws_col = f"p_throws_{p_throws}"
                    if p_throws_col in mlb_features: input_df[p_throws_col] = 1.0
                    stand_col = f"stand_{stand}"
                    if stand_col in mlb_features: input_df[stand_col] = 1.0
                    prev_pitch_col = f"prev_grouped_pitch_{prev_pitch}"
                    if prev_pitch_col in mlb_features: input_df[prev_pitch_col] = 1.0
                    prev_out_col = f"prev_pitch_outcome_{prev_outcome}"
                    if prev_out_col in mlb_features: input_df[prev_out_col] = 1.0

                    dtest = xgb.DMatrix(input_df)
                    probs_array = pitch_model.predict(dtest)[0]
                    final_probs = [float(p * 100) for p in probs_array]
                    st.success("✅ Statcast 大聯盟真實數據分析完成！")
                except Exception as e:
                    st.error(f"⚠️ 預測失敗。錯誤訊息: {e}")

            top_indices = np.argsort(final_probs)[::-1]
            best_idx, second_idx = top_indices[0], top_indices[1]

            UI_NAMES = {'Changeup': '變速/指叉', 'Curveball': '曲球系', 'Fastball_System': '直球系', 'Slider_Cutter': '滑/卡系'}
            best_pitch_raw = final_classes[best_idx]
            best_prob = final_probs[best_idx]
            second_pitch_raw = final_classes[second_idx]
            second_prob = final_probs[second_idx]

        res_c1, res_c2 = st.columns([1, 1])
        with res_c1:
            st.markdown("##### 📊 模型預測機率")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(label="1. 首選預測球種", value=UI_NAMES.get(best_pitch_raw, best_pitch_raw), delta=f"{best_prob:.1f}%")
            with m2:
                st.metric(label="2. 備用引誘球種", value=UI_NAMES.get(second_pitch_raw, second_pitch_raw), delta=f"{second_prob:.1f}%", delta_color="off")
            ui_kit.probability_bars([UI_NAMES.get(c, c) for c in final_classes], final_probs)

        with res_c2:
            st.markdown("##### 🎯 九宮格落點預測")
            fig = ui_kit.draw_strike_zone_plotly(best_pitch_raw.split('_')[0], best_prob)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

elif app_mode == t("mode_obp"):
    if st.button(t("btn_obp"), use_container_width=True, type="primary"):
        if not HAS_XGB or not obp_model:
            st.error("找不到模型檔案，請檢查 data_mlb 資料夾。")
        else:
            with st.spinner("AI 模型運算中..."):
                hist_b_obp = obp_db_dict["b"].get(str(batter_id), 0.315)
                hist_p_obp = obp_db_dict["p"].get(str(pitcher_id), 0.315)

                clean_batter_name = selected_batter.replace("🔥 ", "")
                clean_pitcher_name = selected_pitcher.replace("🔥 ", "")
                st.info(f"數據庫載入成功：{clean_batter_name} OBP ({hist_b_obp:.3f}) / {clean_pitcher_name} 被上壘率 ({hist_p_obp:.3f})")

                feature_names = ['balls', 'strikes', 'outs_when_up', 'inning', 'score_diff',
                                 'runners_on_base', 'pitch_count', 'batter_hist_obp',
                                 'pitcher_hist_obp_allowed', 'is_home_team', 'platoon_advantage', 'base_state_code']

                feature_values = [
                    balls, strikes, outs, inning, score_diff,
                    runners_on_base, pitch_count, hist_b_obp, hist_p_obp,
                    1 if is_home else 0, 1 if platoon else 0, base_state_code
                ]

                try:
                    df_input = pd.DataFrame([feature_values], columns=feature_names)
                    dmatrix = xgb.DMatrix(df_input)
                    prob = obp_model.predict(dmatrix)[0]

                    res_c1, res_c2 = st.columns([1, 1])
                    with res_c1:
                        fig = ui_kit.risk_gauge(float(prob), title=f"{clean_batter_name} 該打席上壘機率 (xOBP)", low_threshold=0.28, high_threshold=0.33)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    with res_c2:
                        st.metric(label=f"預測 {clean_batter_name} 該打席上壘機率 (xOBP)", value=f"{prob:.1%}")
                        if prob > 0.33:
                            st.error("🚨 高上壘風險！")
                        else:
                            st.success("🟢 投手佔優勢。")
                except Exception as e:
                    st.error(f"⚠️ OBP 模型推論失敗。錯誤訊息: {e}")
