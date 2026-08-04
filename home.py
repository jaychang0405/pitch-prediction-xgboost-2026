# -*- coding: utf-8 -*-
import streamlit as st
import ui_kit

st.set_page_config(
    page_title="棒球動態決策支援系統",
    page_icon="data_cpbl/tennis-ball.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ui_kit.inject_theme()
st.logo("data_cpbl/tennis-ball.svg", size="large")

logged_in = bool(st.session_state.get("user"))

pages = [st.Page("views/home_view.py", title="首頁", icon="🏠", url_path="", default=True)]

if logged_in:
    pages += [
        st.Page("views/cpbl_app.py", title="CPBL app", icon="🇹🇼", url_path="CPBL_app"),
        st.Page("views/mlb_app.py", title="MLB app", icon="🇺🇸", url_path="MLB_app"),
        st.Page("views/tunneling_lab.py", title="3D Tunneling Lab", icon="🌀", url_path="3D_Tunneling_Lab"),
    ]
    ui_kit.user_badge(st.session_state["user"])
else:
    pages += [st.Page("views/auth_view.py", title="登入 / Login", icon="🔐", url_path="login")]

pg = st.navigation(pages, position="top")
pg.run()
