import streamlit as st

st.set_page_config(page_title="棒球數據預測中心", page_icon="⚾", layout="centered")

st.title("⚾ 棒球動態決策支援中心")
st.markdown("""
歡迎來到棒球數據預測系統！
這是一個結合 XGBoost 機器學習技術的即時戰術分析工具。

👈 **請從左側選單選擇您要使用的預測系統：**
* **CPBL 中職預測**：包含球種預測與上壘率 (OBP) 分析。
* **MLB 大聯盟預測**：大聯盟投手球種即時預測。
""")