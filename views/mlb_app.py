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

st.logo(os.path.join(DATA_PATH, "mlb_logo.png"), size="large")

# ==========================================
# 1. 三語字典設定
# ==========================================
LANG = {
    "zh": {
        "title": "MLB 動態決策支援系統",
        "subtitle": "基於 Statcast 與 XGBoost 的即時戰術分析",
        "menu": "功能選單",
        "mode_pitch": "🎯 預測下一球球種",
        "mode_obp": "🏃 預測打擊結果 (上壘率)",
        "input_header": "🎮 輸入當下比賽情境",
        "inning": "1. 局數 / Inning",
        "balls": "2. 壞球 / Balls",
        "strikes": "3. 好球 / Strikes",
        "outs": "4. 出局數 / Outs",
        "runners": "5. 壘上跑者",
        "p_throws": "投手慣用手",
        "stand": "打者站位",
        "score_diff": "比分差",
        "pitch_count": "用球數",
        "pitcher": "3. 投手 (Pitcher)",
        "batter": "3. 打者 (Batter)",
        "cap_pitcher": "投手",
        "cap_batter": "打者",
        "obp_header": "OBP 進階特徵 (XGBoost 必備)",
        "is_home": "主場球隊打擊 (Is Home Team)",
        "platoon": "投打對決優勢 (Platoon Adv.)",
        "seq_header": "球種序列特徵",
        "prev_pitch": "前一球配球",
        "prev_outcome": "前一球結果",
        "btn_pitch": "🎯 開始預測球種",
        "btn_obp": "🏃 評估上壘風險 (OBP)",
        "spinner": "AI 模型運算中...",
        "predict_success": "✅ Statcast 大聯盟真實數據分析完成！",
        "predict_error": "⚠️ 預測失敗。錯誤訊息: {e}",
        "result_header": "📊 模型預測機率",
        "top_pick": "1. 首選預測球種",
        "second_pick": "2. 備用引誘球種",
        "strike_zone_header": "🎯 九宮格落點預測",
        "obp_model_missing": "找不到模型檔案，請檢查 data_mlb 資料夾。",
        "obp_info": "數據庫載入成功：{b} OBP ({hb:.3f}) / {p} 被上壘率 ({hp:.3f})",
        "obp_metric_label": "預測 {b} 該打席上壘機率 (xOBP)",
        "obp_high_risk": "🚨 高上壘風險！",
        "obp_low_risk": "🟢 投手佔優勢。",
        "obp_infer_fail": "⚠️ OBP 模型推論失敗。錯誤訊息: {e}",
    },
    "en": {
        "title": "MLB Dynamic Decision Support",
        "subtitle": "Real-time Tactical Analysis based on Statcast & XGBoost",
        "menu": "Menu",
        "mode_pitch": "🎯 Predict Next Pitch",
        "mode_obp": "🏃 Predict At-Bat Outcome (OBP)",
        "input_header": "🎮 Input Current Context",
        "inning": "1. Inning",
        "balls": "2. Balls",
        "strikes": "3. Strikes",
        "outs": "4. Outs",
        "runners": "5. Runners on Base",
        "p_throws": "Pitcher Throws",
        "stand": "Batter Stands",
        "score_diff": "Score Diff",
        "pitch_count": "Pitch Count",
        "pitcher": "3. Pitcher",
        "batter": "3. Batter",
        "cap_pitcher": "Pitcher",
        "cap_batter": "Batter",
        "obp_header": "Advanced OBP Features (XGBoost Required)",
        "is_home": "Home Team Batting",
        "platoon": "Platoon Advantage",
        "seq_header": "Pitch Sequence Features",
        "prev_pitch": "Previous Pitch",
        "prev_outcome": "Previous Pitch Outcome",
        "btn_pitch": "🎯 Predict Pitch",
        "btn_obp": "🏃 Evaluate OBP Risk",
        "spinner": "AI model computing...",
        "predict_success": "✅ Statcast MLB data analysis complete!",
        "predict_error": "⚠️ Prediction failed. Error: {e}",
        "result_header": "📊 Model Prediction",
        "top_pick": "1. Top Predicted Pitch",
        "second_pick": "2. Alternate Setup Pitch",
        "strike_zone_header": "🎯 Strike Zone Prediction",
        "obp_model_missing": "Model file not found. Please check the data_mlb folder.",
        "obp_info": "Loaded: {b} OBP ({hb:.3f}) / {p} OBP allowed ({hp:.3f})",
        "obp_metric_label": "Predicted On-Base Probability (xOBP) for {b}",
        "obp_high_risk": "🚨 High OBP risk!",
        "obp_low_risk": "🟢 Pitcher has the advantage.",
        "obp_infer_fail": "⚠️ OBP inference failed. Error: {e}",
    },
    "ja": {
        "title": "MLB動的意思決定支援システム",
        "subtitle": "StatcastとXGBoostによるリアルタイム戦術分析",
        "menu": "メニュー",
        "mode_pitch": "🎯 次球の球種を予測",
        "mode_obp": "🏃 打席結果を予測（出塁率）",
        "input_header": "🎮 現在の試合状況を入力",
        "inning": "1. 回 / Inning",
        "balls": "2. ボール",
        "strikes": "3. ストライク",
        "outs": "4. アウト",
        "runners": "5. 走者状況",
        "p_throws": "投手の利き腕",
        "stand": "打者の打席",
        "score_diff": "得点差",
        "pitch_count": "球数",
        "pitcher": "3. 投手",
        "batter": "3. 打者",
        "cap_pitcher": "投手",
        "cap_batter": "打者",
        "obp_header": "高度なOBP特徴（XGBoost必須）",
        "is_home": "ホームチーム打撃",
        "platoon": "投打相性優位",
        "seq_header": "球種シーケンス特徴",
        "prev_pitch": "前球の配球",
        "prev_outcome": "前球の結果",
        "btn_pitch": "🎯 球種を予測開始",
        "btn_obp": "🏃 出塁リスクを評価（OBP）",
        "spinner": "AIモデル計算中...",
        "predict_success": "✅ Statcastデータ分析が完了しました！",
        "predict_error": "⚠️ 予測に失敗しました。エラー: {e}",
        "result_header": "📊 モデル予測確率",
        "top_pick": "1. 第一予測球種",
        "second_pick": "2. 誘い球候補",
        "strike_zone_header": "🎯 ナインゾーン予測",
        "obp_model_missing": "モデルファイルが見つかりません。data_mlb フォルダを確認してください。",
        "obp_info": "読込完了：{b} OBP ({hb:.3f}) / {p} 被OBP ({hp:.3f})",
        "obp_metric_label": "{b} のこの打席の出塁確率予測 (xOBP)",
        "obp_high_risk": "🚨 出塁リスク高！",
        "obp_low_risk": "🟢 投手有利。",
        "obp_infer_fail": "⚠️ OBPモデルの推論に失敗しました。エラー: {e}",
    },
}

BASE_KEYS = ["1B", "2B", "3B"]
BASE_LABELS = {
    "zh": {"1B": "一壘 (1B)", "2B": "二壘 (2B)", "3B": "三壘 (3B)"},
    "en": {"1B": "1st Base", "2B": "2nd Base", "3B": "3rd Base"},
    "ja": {"1B": "一塁", "2B": "二塁", "3B": "三塁"},
}
PREV_PITCH_KEYS = ["First_Pitch", "Fastball_System", "Slider_Cutter", "Curveball", "Changeup"]
PREV_PITCH_LABELS = {
    "zh": {"First_Pitch": "首球", "Fastball_System": "直球系", "Slider_Cutter": "滑/卡系", "Curveball": "曲球", "Changeup": "變速/指叉系"},
    "en": {"First_Pitch": "First Pitch", "Fastball_System": "Fastball", "Slider_Cutter": "Slider/Cutter", "Curveball": "Curveball", "Changeup": "Changeup"},
    "ja": {"First_Pitch": "初球", "Fastball_System": "直球系", "Slider_Cutter": "スライダー/カット系", "Curveball": "カーブ", "Changeup": "チェンジアップ系"},
}
PREV_OUTCOME_KEYS = ["First_Pitch", "ball", "called_strike", "swinging_strike", "foul", "hit_into_play", "other"]
PREV_OUTCOME_LABELS = {
    "zh": {"First_Pitch": "首球", "ball": "壞球", "called_strike": "看好球", "swinging_strike": "揮空", "foul": "界外", "hit_into_play": "打進場內", "other": "其他"},
    "en": {"First_Pitch": "First Pitch", "ball": "Ball", "called_strike": "Called Strike", "swinging_strike": "Swinging Strike", "foul": "Foul", "hit_into_play": "In-Play", "other": "Other"},
    "ja": {"First_Pitch": "初球", "ball": "ボール", "called_strike": "見逃しストライク", "swinging_strike": "空振り", "foul": "ファウル", "hit_into_play": "インプレー", "other": "その他"},
}
PITCH_CLASSES = ['Changeup', 'Curveball', 'Fastball_System', 'Slider_Cutter']
UI_NAMES = {
    "zh": {'Changeup': '變速/指叉', 'Curveball': '曲球系', 'Fastball_System': '直球系', 'Slider_Cutter': '滑/卡系'},
    "en": {'Changeup': 'Changeup', 'Curveball': 'Curveball', 'Fastball_System': 'Fastball', 'Slider_Cutter': 'Slider/Cutter'},
    "ja": {'Changeup': 'チェンジアップ系', 'Curveball': 'カーブ系', 'Fastball_System': '直球系', 'Slider_Cutter': 'スライダー/カット系'},
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
l = ui_kit.language_switcher()
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
        with c1: inning = st.number_input(t("inning"), 1, 12, 1)
        with c2: balls = st.selectbox(t("balls"), [0, 1, 2, 3])
        with c3: strikes = st.selectbox(t("strikes"), [0, 1, 2])
        with c4: outs = st.selectbox(t("outs"), [0, 1, 2])

        c_base, c_pd1, c_pd2 = st.columns([2, 1, 1])
        with c_base:
            bases = st.multiselect(t("runners"), BASE_KEYS, format_func=lambda k: BASE_LABELS[l][k])
            on_1b = 1 if "1B" in bases else 0
            on_2b = 1 if "2B" in bases else 0
            on_3b = 1 if "3B" in bases else 0
            runners_on_base = on_1b + on_2b + on_3b
            base_state_code = (on_1b * 1) + (on_2b * 2) + (on_3b * 4)

        with c_pd1:
            if app_mode == t("mode_pitch"): p_throws = st.selectbox(t("p_throws"), ["R", "L"])
            else: score_diff = st.number_input(t("score_diff"), value=0)
        with c_pd2:
            if app_mode == t("mode_pitch"): stand = st.selectbox(t("stand"), ["R", "L"])
            else: pitch_count = st.number_input(t("pitch_count"), 1, 150, 15)

        c5, c6 = st.columns(2)
        with c5: selected_pitcher = st.selectbox(t("pitcher"), list(pitchers_db.keys()))
        with c6: selected_batter = st.selectbox(t("batter"), list(batters_db.keys()))

        if app_mode == t("mode_obp"):
            st.markdown(f"#### {t('obp_header')}")
            o1, o2 = st.columns(2)
            with o1: is_home = st.toggle(t("is_home"), value=True)
            with o2: platoon = st.toggle(t("platoon"), value=False)

        if app_mode == t("mode_pitch"):
            st.markdown(f"#### {t('seq_header')}")
            c7, c8 = st.columns(2)
            with c7:
                prev_pitch = st.selectbox(t("prev_pitch"), PREV_PITCH_KEYS, format_func=lambda k: PREV_PITCH_LABELS[l][k])
            with c8:
                prev_outcome = st.selectbox(t("prev_outcome"), PREV_OUTCOME_KEYS, format_func=lambda k: PREV_OUTCOME_LABELS[l][k])

    pitcher_id = pitchers_db.get(selected_pitcher, 660271)
    batter_id = batters_db.get(selected_batter, 660271)

    with col_avatars:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1: st.image(get_headshot_url(pitcher_id), caption=t("cap_pitcher"), width=120)
        with a2: st.image(get_headshot_url(batter_id), caption=t("cap_batter"), width=120)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. 真實 XGBoost 預測邏輯 (保持不變)
# ==========================================
if app_mode == t("mode_pitch"):
    if st.button(t("btn_pitch"), use_container_width=True, type="primary"):
        with st.spinner(t("spinner")):
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
                    st.success(t("predict_success"))
                except Exception as e:
                    st.error(t("predict_error").format(e=e))

            top_indices = np.argsort(final_probs)[::-1]
            best_idx, second_idx = top_indices[0], top_indices[1]

            names = UI_NAMES[l]
            best_pitch_raw = final_classes[best_idx]
            best_prob = final_probs[best_idx]
            second_pitch_raw = final_classes[second_idx]
            second_prob = final_probs[second_idx]

        res_c1, res_c2 = st.columns([1, 1])
        with res_c1:
            st.markdown(f"##### {t('result_header')}")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(label=t("top_pick"), value=names.get(best_pitch_raw, best_pitch_raw), delta=f"{best_prob:.1f}%")
            with m2:
                st.metric(label=t("second_pick"), value=names.get(second_pitch_raw, second_pitch_raw), delta=f"{second_prob:.1f}%", delta_color="off")
            ui_kit.probability_bars([names.get(c, c) for c in final_classes], final_probs)

        with res_c2:
            st.markdown(f"##### {t('strike_zone_header')}")
            fig = ui_kit.draw_strike_zone_plotly(best_pitch_raw.split('_')[0], best_prob, lang=l)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

elif app_mode == t("mode_obp"):
    if st.button(t("btn_obp"), use_container_width=True, type="primary"):
        if not HAS_XGB or not obp_model:
            st.error(t("obp_model_missing"))
        else:
            with st.spinner(t("spinner")):
                hist_b_obp = obp_db_dict["b"].get(str(batter_id), 0.315)
                hist_p_obp = obp_db_dict["p"].get(str(pitcher_id), 0.315)

                clean_batter_name = selected_batter.replace("🔥 ", "")
                clean_pitcher_name = selected_pitcher.replace("🔥 ", "")
                st.info(t("obp_info").format(b=clean_batter_name, hb=hist_b_obp, p=clean_pitcher_name, hp=hist_p_obp))

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
                        fig = ui_kit.risk_gauge(float(prob), title=t("obp_metric_label").format(b=clean_batter_name), low_threshold=0.28, high_threshold=0.33)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    with res_c2:
                        st.metric(label=t("obp_metric_label").format(b=clean_batter_name), value=f"{prob:.1%}")
                        if prob > 0.33:
                            st.error(t("obp_high_risk"))
                        else:
                            st.success(t("obp_low_risk"))
                except Exception as e:
                    st.error(t("obp_infer_fail").format(e=e))
