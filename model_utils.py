# -*- coding: utf-8 -*-
"""共用資料/模型載入邏輯：不依賴 streamlit，供 views/mlb_app.py 與
views/cpbl_app.py 的 @st.cache_resource / @st.cache_data wrapper 呼叫。
避免同一段模型載入或特徵組裝程式碼在兩個 app 各存一份、改一邊忘了改另一邊。
"""
import os
import json
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DEFAULT_ENCODINGS = ['utf-8', 'utf-8-sig', 'big5', 'cp950', 'ansi']


def load_xgb_booster(path):
    """載入 XGBoost Booster；檔案不存在或未安裝 xgboost 時回傳 None。"""
    if not HAS_XGB or not os.path.exists(path):
        return None
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def safe_read_csv(path, encodings=DEFAULT_ENCODINGS):
    """依序嘗試多種編碼讀取 CSV，全部失敗（含檔案不存在）則回傳空 DataFrame。"""
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()


def safe_read_json(path, encodings=DEFAULT_ENCODINGS):
    """依序嘗試多種編碼讀取 JSON，全部失敗（含檔案不存在）則回傳空 list。"""
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    return []


# OBP 模型的特徵欄位與順序：MLB 與 CPBL 兩個 app 餵給同一種 xgb_obp 模型結構，
# 欄位順序必須跟訓練時完全一致，因此只在這裡定義一次，避免兩邊各存一份、
# 未來調整時漏改其中一邊導致特徵錯位卻不會報錯。
OBP_FEATURE_NAMES = [
    'balls', 'strikes', 'outs_when_up', 'inning', 'score_diff',
    'runners_on_base', 'pitch_count', 'batter_hist_obp',
    'pitcher_hist_obp_allowed', 'is_home_team', 'platoon_advantage', 'base_state_code',
]


def build_obp_features(balls, strikes, outs, inning, score_diff, runners_on_base,
                        pitch_count, hist_b_obp, hist_p_obp, is_home, platoon, base_state_code):
    """依 OBP_FEATURE_NAMES 的順序組成單列 DataFrame，可直接丟進 xgb.DMatrix。"""
    values = [
        balls, strikes, outs, inning, score_diff, runners_on_base,
        pitch_count, hist_b_obp, hist_p_obp,
        1 if is_home else 0, 1 if platoon else 0, base_state_code,
    ]
    return pd.DataFrame([values], columns=OBP_FEATURE_NAMES)
