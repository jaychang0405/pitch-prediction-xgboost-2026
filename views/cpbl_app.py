# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import ui_kit
import cpbl_live

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_PATH = "data_cpbl"  # 設定為獨立資料夾

st.logo(os.path.join(DATA_PATH, "cpbl_icon.png"), size="large")

# ==========================================
# 0. 密碼鎖與路徑設定
# ==========================================
def clean_name(name):
    """輔助函式：去除掉下拉選單中為了美觀加上的火焰符號"""
    if isinstance(name, str):
        return name.replace("🔥 ", "").strip()
    return name

# ==========================================
# 1. 三語字典設定
# ==========================================
LANG = {
    "zh": {
        "title": "中職動態決策支援系統",
        "subtitle": "基於 XGBoost 的球種預測與上壘率 (OBP) 分析",
        "menu": "功能選單",
        "mode_pitch": "🎯 預測下一球球種",
        "mode_obp": "🏃 預測打擊結果 (上壘率)",
        "input_header": "🎮 輸入當下比賽情境",
        "inning": "1. 局數 / Inning",
        "balls": "2. 壞球 / Balls",
        "strikes": "3. 好球 / Strikes",
        "outs": "4. 出局數 / Outs",
        "pitcher": "投手 (Pitcher)",
        "batter": "打者 (Batter)",
        "no_data": "無資料",
        "seq_header": "球種序列特徵",
        "prev_pitch": "前一球球種",
        "prev_outcome": "前一球結果",
        "obp_header": "模型進階特徵 (XGBoost 必備欄位)",
        "score_diff": "比分差 (主隊減客隊，領先為正)",
        "base_state": "壘上局面",
        "pitch_count": "投手當前用球數",
        "is_home": "主場球隊打擊 (Is Home Team)",
        "platoon": "投打對決優勢 (Platoon Adv.)",
        "platoon_help": "若右打對左投，或左打對右投，請開啟此項",
        "btn_pitch": "🎯 開始預測球種與位置",
        "btn_obp": "🏃 開始評估上壘風險",
        "spinner": "AI 模型運算中...",
        "analysis_done": "✅ 分析完成！投手：{p} vs 打者：{b}",
        "predict_warning": "⚠️ 預測發生錯誤，顯示模擬數據。錯誤: {e}",
        "result_header": "📊 模型預測數據",
        "top_pick": "1. 首選建議球種",
        "second_pick": "2. 備用引誘球種",
        "strike_zone_header": "🎯 九宮格視覺化 (Strike Zone)",
        "obp_model_missing": "找不到模型檔案，請確認 `cpbl_obp_model.json` 已放置於 data_cpbl 資料夾中。",
        "obp_info": "系統自動偵測帶入：{b} 歷史 OBP ({hb:.3f}) / {p} 歷史被 OBP ({hp:.3f})",
        "obp_metric_label": "預測 {b} 該打席上壘機率 (xOBP)",
        "obp_high_risk": "⚠️ 高上壘風險！打者具備優勢，建議投手採取邊角引誘球策略。",
        "obp_low_risk": "✅ 目前對戰情境對投手有利。",
        "obp_infer_fail": "⚠️ OBP 模型推論失敗，請檢查資料格式。錯誤: {e}",
        "live_header": "🏆 即時戰績與排行榜",
        "live_subtitle": "資料來源：CPBL 官方網站，每 30 分鐘更新一次",
        "standings_header": "球隊戰績",
        "toplist_pitching_header": "⚾ 投手 TOP5",
        "toplist_batting_header": "🏏 打者 TOP5",
        "col_rank": "排名",
        "col_team": "球隊",
        "col_games": "出賽",
        "col_record": "勝-和-敗",
        "col_win_pct": "勝率",
        "col_games_behind": "勝差",
        "col_home": "主場戰績",
        "col_away": "客場戰績",
        "col_streak": "連勝/連敗",
        "col_last10": "近十場",
        "live_unavailable": "⚠️ 目前無法取得 CPBL 官網即時數據，請稍後再試。",
    },
    "en": {
        "title": "CPBL Dynamic Decision Support",
        "subtitle": "Pitch Prediction & OBP Analysis based on XGBoost",
        "menu": "Menu",
        "mode_pitch": "🎯 Predict Next Pitch",
        "mode_obp": "🏃 Predict At-Bat Outcome (OBP)",
        "input_header": "🎮 Input Current Context",
        "inning": "1. Inning",
        "balls": "2. Balls",
        "strikes": "3. Strikes",
        "outs": "4. Outs",
        "pitcher": "Pitcher",
        "batter": "Batter",
        "no_data": "No data",
        "seq_header": "Pitch Sequence Features",
        "prev_pitch": "Previous Pitch Type",
        "prev_outcome": "Previous Pitch Outcome",
        "obp_header": "Advanced Model Features (XGBoost Required)",
        "score_diff": "Score Diff (Home - Away, positive = leading)",
        "base_state": "Base State",
        "pitch_count": "Pitcher's Current Pitch Count",
        "is_home": "Home Team Batting",
        "platoon": "Platoon Advantage",
        "platoon_help": "Enable if right batter vs. left pitcher, or left batter vs. right pitcher",
        "btn_pitch": "🎯 Predict Pitch & Location",
        "btn_obp": "🏃 Evaluate OBP Risk",
        "spinner": "AI model computing...",
        "analysis_done": "✅ Analysis complete! Pitcher: {p} vs Batter: {b}",
        "predict_warning": "⚠️ Prediction error, showing simulated data. Error: {e}",
        "result_header": "📊 Model Prediction",
        "top_pick": "1. Top Suggested Pitch",
        "second_pick": "2. Alternate Setup Pitch",
        "strike_zone_header": "🎯 Strike Zone Visualization",
        "obp_model_missing": "Model file not found. Please make sure `cpbl_obp_model.json` is in the data_cpbl folder.",
        "obp_info": "Auto-loaded: {b} historical OBP ({hb:.3f}) / {p} historical OBP allowed ({hp:.3f})",
        "obp_metric_label": "Predicted On-Base Probability (xOBP) for {b}",
        "obp_high_risk": "⚠️ High OBP risk! The batter has the advantage — pitcher should work the edges.",
        "obp_low_risk": "✅ The current matchup favors the pitcher.",
        "obp_infer_fail": "⚠️ OBP inference failed, please check the data format. Error: {e}",
        "live_header": "🏆 Live Standings & Leaderboards",
        "live_subtitle": "Source: CPBL official website, refreshed every 30 minutes",
        "standings_header": "Team Standings",
        "toplist_pitching_header": "⚾ Pitching Leaders (Top 5)",
        "toplist_batting_header": "🏏 Batting Leaders (Top 5)",
        "col_rank": "Rank",
        "col_team": "Team",
        "col_games": "G",
        "col_record": "W-D-L",
        "col_win_pct": "Win%",
        "col_games_behind": "GB",
        "col_home": "Home",
        "col_away": "Away",
        "col_streak": "Streak",
        "col_last10": "Last 10",
        "live_unavailable": "⚠️ Live CPBL data is unavailable right now, please try again later.",
    },
    "ja": {
        "title": "CPBL動的意思決定支援システム",
        "subtitle": "XGBoostによる球種予測と出塁率 (OBP) 分析",
        "menu": "メニュー",
        "mode_pitch": "🎯 次球の球種を予測",
        "mode_obp": "🏃 打席結果を予測（出塁率）",
        "input_header": "🎮 現在の試合状況を入力",
        "inning": "1. 回 / Inning",
        "balls": "2. ボール",
        "strikes": "3. ストライク",
        "outs": "4. アウト",
        "pitcher": "投手",
        "batter": "打者",
        "no_data": "データなし",
        "seq_header": "球種シーケンス特徴",
        "prev_pitch": "前の球種",
        "prev_outcome": "前球の結果",
        "obp_header": "高度なモデル特徴（XGBoost必須項目）",
        "score_diff": "得点差（主催チーム-相手、リードでプラス）",
        "base_state": "走者状況",
        "pitch_count": "投手の現在の球数",
        "is_home": "ホームチーム打撃",
        "platoon": "投打相性優位",
        "platoon_help": "右打者対左投手、または左打者対右投手の場合はオンにしてください",
        "btn_pitch": "🎯 球種と位置を予測開始",
        "btn_obp": "🏃 出塁リスクを評価開始",
        "spinner": "AIモデル計算中...",
        "analysis_done": "✅ 分析完了！投手：{p} vs 打者：{b}",
        "predict_warning": "⚠️ 予測エラーが発生、シミュレーションデータを表示します。エラー: {e}",
        "result_header": "📊 モデル予測データ",
        "top_pick": "1. 第一候補球種",
        "second_pick": "2. 誘い球候補",
        "strike_zone_header": "🎯 ナインゾーン可視化 (Strike Zone)",
        "obp_model_missing": "モデルファイルが見つかりません。`cpbl_obp_model.json` が data_cpbl フォルダにあるか確認してください。",
        "obp_info": "システム自動読込：{b} 過去OBP ({hb:.3f}) / {p} 被OBP ({hp:.3f})",
        "obp_metric_label": "{b} のこの打席の出塁確率予測 (xOBP)",
        "obp_high_risk": "⚠️ 出塁リスク高！打者有利のため、投手はコーナーワークを推奨。",
        "obp_low_risk": "✅ 現在の対戦状況は投手有利です。",
        "obp_infer_fail": "⚠️ OBPモデルの推論に失敗しました。データ形式を確認してください。エラー: {e}",
        "live_header": "🏆 リアルタイム順位・ランキング",
        "live_subtitle": "データ提供：CPBL公式サイト（30分ごとに更新）",
        "standings_header": "チーム順位",
        "toplist_pitching_header": "⚾ 投手成績 TOP5",
        "toplist_batting_header": "🏏 打者成績 TOP5",
        "col_rank": "順位",
        "col_team": "チーム",
        "col_games": "試合数",
        "col_record": "勝-分-敗",
        "col_win_pct": "勝率",
        "col_games_behind": "差",
        "col_home": "本拠地",
        "col_away": "ビジター",
        "col_streak": "連勝/連敗",
        "col_last10": "直近10試合",
        "live_unavailable": "⚠️ 現在CPBLの最新データを取得できません。後でもう一度お試しください。",
    },
}

PITCH_SEQ_KEYS = ["First_Pitch", "Fastball_System", "Slider_Cutter", "Curveball", "Changeup"]
PITCH_SEQ_LABELS = {
    "zh": {"First_Pitch": "首球", "Fastball_System": "直球系", "Slider_Cutter": "滑/卡系", "Curveball": "曲球", "Changeup": "變速/指叉系"},
    "en": {"First_Pitch": "First Pitch", "Fastball_System": "Fastball", "Slider_Cutter": "Slider/Cutter", "Curveball": "Curveball", "Changeup": "Changeup"},
    "ja": {"First_Pitch": "初球", "Fastball_System": "直球系", "Slider_Cutter": "スライダー/カット系", "Curveball": "カーブ", "Changeup": "チェンジアップ系"},
}
PITCH_OUTCOME_KEYS = ["First_Pitch", "Ball", "Strike", "Foul", "In-Play"]
PITCH_OUTCOME_LABELS = {
    "zh": {"First_Pitch": "首球", "Ball": "壞球", "Strike": "好球", "Foul": "界外", "In-Play": "打進場內"},
    "en": {"First_Pitch": "First Pitch", "Ball": "Ball", "Strike": "Strike", "Foul": "Foul", "In-Play": "In-Play"},
    "ja": {"First_Pitch": "初球", "Ball": "ボール", "Strike": "ストライク", "Foul": "ファウル", "In-Play": "インプレー"},
}
PITCH_CLASSES = ['Changeup', 'Curveball', 'Fastball_System', 'Slider_Cutter']
PITCH_TRANS = {
    "zh": {'Changeup': '變速/指叉系', 'Curveball': '曲球', 'Fastball_System': '直球系', 'Slider_Cutter': '滑/卡系'},
    "en": {'Changeup': 'Changeup', 'Curveball': 'Curveball', 'Fastball_System': 'Fastball', 'Slider_Cutter': 'Slider/Cutter'},
    "ja": {'Changeup': 'チェンジアップ系', 'Curveball': 'カーブ', 'Fastball_System': '直球系', 'Slider_Cutter': 'スライダー/カット系'},
}
BASE_STATE_KEYS = ["none", "1b", "2b", "1b_2b", "3b", "1b_3b", "2b_3b", "loaded"]
BASE_STATE_VALUES = {
    "none": (0, 0), "1b": (1, 1), "2b": (1, 2), "1b_2b": (2, 3),
    "3b": (1, 4), "1b_3b": (2, 5), "2b_3b": (2, 6), "loaded": (3, 7),
}
BASE_STATE_LABELS = {
    "zh": {"none": "無人在壘", "1b": "一壘有人", "2b": "二壘有人", "1b_2b": "一二壘有人",
           "3b": "三壘有人", "1b_3b": "一三壘有人", "2b_3b": "二三壘有人", "loaded": "滿壘"},
    "en": {"none": "Bases Empty", "1b": "Runner on 1B", "2b": "Runner on 2B", "1b_2b": "Runners on 1B & 2B",
           "3b": "Runner on 3B", "1b_3b": "Runners on 1B & 3B", "2b_3b": "Runners on 2B & 3B", "loaded": "Bases Loaded"},
    "ja": {"none": "走者なし", "1b": "一塁走者", "2b": "二塁走者", "1b_2b": "一・二塁走者",
           "3b": "三塁走者", "1b_3b": "一・三塁走者", "2b_3b": "二・三塁走者", "loaded": "満塁"},
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
# 2b. CPBL 官網即時戰績與排行榜
# ==========================================
@st.cache_data(ttl=1800, show_spinner=False)
def cached_toplist():
    return cpbl_live.fetch_toplist()

@st.cache_data(ttl=1800, show_spinner=False)
def cached_standings():
    return cpbl_live.fetch_standings()

def render_leader_card(category, t):
    leaders = category["leaders"]
    if not leaders:
        return
    top = leaders[0]
    rest_html = "".join(
        f'<div style="text-align:left; font-size:0.82rem; padding:0.1rem 0;">'
        f'{l["rank"]}. {l["name"]} <span style="opacity:0.6;">({l["team"]})</span> '
        f'<b style="float:right;">{l["value"]}</b></div>'
        for l in leaders[1:]
    )
    photo_html = (
        f'<img class="uikit-player-avatar" style="width:64px;height:64px;" src="{category["photo_url"]}" />'
        if category.get("photo_url") else "🏅"
    )
    st.markdown(
        f"""
        <div class="uikit-card" style="text-align:center; padding:1rem 0.9rem;">
            <div style="font-weight:800;">{category['title_zh']} <span style="opacity:0.55; font-size:0.82rem;">{category['title_en']}</span></div>
            <div style="margin:0.5rem 0;">{photo_html}</div>
            <div style="font-weight:700;">{top['name']}</div>
            <div style="font-size:0.78rem; opacity:0.6;">{top['team']}</div>
            <div style="font-size:1.35rem; font-weight:800; color:{ui_kit.ACCENT_BLUE}; margin:0.25rem 0 0.6rem;">{top['value']}</div>
            <hr style="opacity:0.12; margin:0.4rem 0;" />
            {rest_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_live_section(t):
    st.markdown(f"### {t('live_header')}")
    st.caption(t("live_subtitle"))

    standings = cached_standings()
    if standings is None:
        st.warning(t("live_unavailable"))
    else:
        df = pd.DataFrame(standings)
        df = df.rename(columns={
            "rank": t("col_rank"), "team": t("col_team"), "games": t("col_games"),
            "record": t("col_record"), "win_pct": t("col_win_pct"), "games_behind": t("col_games_behind"),
            "home_record": t("col_home"), "away_record": t("col_away"), "streak": t("col_streak"), "last10": t("col_last10"),
        })
        st.markdown(f"#### {t('standings_header')}")
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_order=[t("col_rank"), "logo_url", t("col_team"), t("col_games"), t("col_record"),
                          t("col_win_pct"), t("col_games_behind"), t("col_home"), t("col_away"),
                          t("col_streak"), t("col_last10")],
            column_config={
                "logo_url": st.column_config.ImageColumn(label=""),
            },
        )

    st.markdown("<br>", unsafe_allow_html=True)
    pitching, batting = cached_toplist()
    if pitching is None or batting is None:
        st.warning(t("live_unavailable"))
    else:
        st.markdown(f"#### {t('toplist_pitching_header')}")
        cols = st.columns(len(pitching))
        for col, cat in zip(cols, pitching):
            with col:
                render_leader_card(cat, t)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### {t('toplist_batting_header')}")
        cols = st.columns(len(batting))
        for col, cat in zip(cols, batting):
            with col:
                render_leader_card(cat, t)

    st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 3. 頁面標題列 (含右上角語言切換)
# ==========================================
l = ui_kit.language_switcher()
def t(key): return LANG[l].get(key, key)

ui_kit.hero_banner(t("title"), t("subtitle"), icon="🇹🇼")

render_live_section(t)

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
# 5. 共用情境輸入區塊
# ==========================================
st.header(t("input_header"))

if app_mode == t("mode_pitch"):
    p_names = data["pitch"]["p"]["player_name"].tolist() if "p" in data["pitch"] else [t("no_data")]
    b_names = data["pitch"]["b"]["player_name"].tolist() if "b" in data["pitch"] else [t("no_data")]
else:
    p_names = data["obp_list"]["p"]["player_name"].tolist() if "p" in data["obp_list"] else [t("no_data")]
    b_names = data["obp_list"]["b"]["player_name"].tolist() if "b" in data["obp_list"] else [t("no_data")]

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        inning = st.number_input(t("inning"), 1, 12, 1)
    with c2:
        balls = st.selectbox(t("balls"), [0, 1, 2, 3])
    with c3:
        strikes = st.selectbox(t("strikes"), [0, 1, 2])
    with c4:
        outs = st.selectbox(t("outs"), [0, 1, 2])

    c5, c6 = st.columns(2)
    with c5:
        selected_pitcher = st.selectbox(t("pitcher"), p_names)
        clean_pitcher = clean_name(selected_pitcher)
    with c6:
        selected_batter = st.selectbox(t("batter"), b_names)
        clean_batter = clean_name(selected_batter)

    # ------------------------------------------
    # 模式 A: 球種預測 - 額外欄位
    # ------------------------------------------
    if app_mode == t("mode_pitch"):
        st.markdown(f"#### {t('seq_header')}")
        c7, c8 = st.columns(2)
        with c7:
            prev_pitch = st.selectbox(t("prev_pitch"), PITCH_SEQ_KEYS, format_func=lambda k: PITCH_SEQ_LABELS[l][k])
        with c8:
            prev_outcome = st.selectbox(t("prev_outcome"), PITCH_OUTCOME_KEYS, format_func=lambda k: PITCH_OUTCOME_LABELS[l][k])

    # ------------------------------------------
    # 模式 B: 上壘率預測 - 額外欄位
    # ------------------------------------------
    elif app_mode == t("mode_obp"):
        st.markdown(f"#### {t('obp_header')}")
        r1, r2, r3 = st.columns(3)
        with r1:
            score_diff = st.number_input(t("score_diff"), value=0)
        with r2:
            selected_base = st.selectbox(t("base_state"), BASE_STATE_KEYS, format_func=lambda k: BASE_STATE_LABELS[l][k])
            runners_on_base, base_state_code = BASE_STATE_VALUES[selected_base]
        with r3:
            pitch_count = st.number_input(t("pitch_count"), 0, 150, 15)

        r4, r5 = st.columns(2)
        with r4:
            is_home = st.toggle(t("is_home"), value=True)
        with r5:
            platoon = st.toggle(t("platoon"), value=False, help=t("platoon_help"))

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. 預測邏輯 (Pitch & OBP)
# ==========================================

# ------------------------------------------
# 模式 A: 球種預測
# ------------------------------------------
if app_mode == t("mode_pitch"):
    if st.button(t("btn_pitch"), use_container_width=True, type="primary"):
        with st.spinner(t("spinner")):
            st.success(t("analysis_done").format(p=clean_pitcher, b=clean_batter))

            trans = PITCH_TRANS[l]
            ui_predicted_name, ui_secondary_name = trans['Fastball_System'], trans['Changeup']
            plot_predicted_name_en, predicted_prob, secondary_prob = "Fastball", 52.5, 28.3
            chart_names = [ui_predicted_name, ui_secondary_name, trans['Slider_Cutter'], trans['Curveball']]
            chart_probs = [predicted_prob, secondary_prob, 14.2, 5.0]

            if HAS_XGB and pitch_model and data["features"]:
                try:
                    expected_features = data["features"]

                    input_df = pd.DataFrame(0.0, index=[0], columns=expected_features)
                    if 'outs' in expected_features: input_df['outs'] = outs
                    if 'pitch_count' in expected_features: input_df['pitch_count'] = 15

                    count_col = f"count_{balls}-{strikes}"
                    if count_col in expected_features: input_df[count_col] = 1.0

                    prev_pitch_col = f"prev_grouped_pitch_{prev_pitch}"
                    if prev_pitch_col in expected_features: input_df[prev_pitch_col] = 1.0

                    dtest = xgb.DMatrix(input_df.to_numpy(dtype='float32'))
                    probs = pitch_model.predict(dtest)[0]

                    top_indices = np.argsort(probs)[::-1]
                    best_idx, second_idx = top_indices[0], top_indices[1]

                    ui_predicted_name = trans[PITCH_CLASSES[best_idx]]
                    ui_secondary_name = trans[PITCH_CLASSES[second_idx]]
                    plot_predicted_name_en = PITCH_CLASSES[best_idx].split('_')[0]
                    predicted_prob = float(probs[best_idx] * 100)
                    secondary_prob = float(probs[second_idx] * 100)
                    chart_probs = [float(p * 100) for p in probs]
                    chart_names = [trans[c] for c in PITCH_CLASSES]

                except Exception as e:
                    st.warning(t("predict_warning").format(e=e))

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.markdown(f"##### {t('result_header')}")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(label=t("top_pick"), value=ui_predicted_name, delta=f"{predicted_prob:.1f}%")
            with m2:
                st.metric(label=t("second_pick"), value=ui_secondary_name, delta=f"{secondary_prob:.1f}%", delta_color="off")
            ui_kit.probability_bars(chart_names, chart_probs)
        with res_col2:
            st.markdown(f"##### {t('strike_zone_header')}")
            fig = ui_kit.draw_strike_zone_plotly(plot_predicted_name_en, predicted_prob, lang=l)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ------------------------------------------
# 模式 B: 上壘率預測
# ------------------------------------------
elif app_mode == t("mode_obp"):
    if st.button(t("btn_obp"), use_container_width=True, type="primary"):
        if obp_model is None:
            st.error(t("obp_model_missing"))
        else:
            with st.spinner(t("spinner")):
                hist_b_obp = data["obp_db_dict"]["b"].get(clean_batter, 0.330)
                hist_p_obp = data["obp_db_dict"]["p"].get(clean_pitcher, 0.330)

                st.info(t("obp_info").format(b=clean_batter, hb=hist_b_obp, p=clean_pitcher, hp=hist_p_obp))

                feature_names = ['balls', 'strikes', 'outs_when_up', 'inning', 'score_diff', 'runners_on_base',
                                 'pitch_count', 'batter_hist_obp', 'pitcher_hist_obp_allowed', 'is_home_team',
                                 'platoon_advantage', 'base_state_code']

                feature_values = [
                    balls, strikes, outs, inning, score_diff, runners_on_base,
                    pitch_count, hist_b_obp, hist_p_obp, 1 if is_home else 0, 1 if platoon else 0, base_state_code
                ]

                try:
                    df_input = pd.DataFrame([feature_values], columns=feature_names)
                    dmatrix = xgb.DMatrix(df_input)
                    prob = obp_model.predict(dmatrix)[0]

                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        fig = ui_kit.risk_gauge(float(prob), title=t("obp_metric_label").format(b=clean_batter))
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    with res_col2:
                        st.metric(label=t("obp_metric_label").format(b=clean_batter), value=f"{prob:.1%}")
                        if prob > 0.35:
                            st.warning(t("obp_high_risk"))
                        else:
                            st.success(t("obp_low_risk"))
                except Exception as e:
                    st.error(t("obp_infer_fail").format(e=e))
