# -*- coding: utf-8 -*-
import streamlit as st
import ui_kit
import auth

LANG = {
    "zh": {
        "title": "會員登入",
        "subtitle": "登入或註冊帳號，即可使用 CPBL / MLB / 3D 預測模組",
        "tab_login": "登入",
        "tab_register": "註冊新帳號",
        "username": "帳號",
        "password": "密碼",
        "confirm_password": "確認密碼",
        "btn_login": "登入",
        "btn_register": "註冊",
        "err_invalid": "⚠️ 帳號或密碼錯誤。",
        "err_empty": "⚠️ 帳號與密碼不可空白。",
        "err_exists": "⚠️ 此帳號已被註冊，請換一個帳號名稱。",
        "err_mismatch": "⚠️ 兩次輸入的密碼不一致。",
        "success_login": "✅ 登入成功！",
        "success_register": "✅ 註冊成功！請切換到「登入」分頁使用新帳號登入。",
    },
    "en": {
        "title": "Member Login",
        "subtitle": "Log in or create an account to access the CPBL / MLB / 3D prediction modules",
        "tab_login": "Log In",
        "tab_register": "Register",
        "username": "Username",
        "password": "Password",
        "confirm_password": "Confirm Password",
        "btn_login": "Log In",
        "btn_register": "Register",
        "err_invalid": "⚠️ Incorrect username or password.",
        "err_empty": "⚠️ Username and password cannot be empty.",
        "err_exists": "⚠️ This username is already taken. Please choose another.",
        "err_mismatch": "⚠️ The two passwords do not match.",
        "success_login": "✅ Logged in successfully!",
        "success_register": "✅ Registration complete! Switch to the \"Log In\" tab to sign in.",
    },
    "ja": {
        "title": "会員ログイン",
        "subtitle": "ログインまたは新規登録すると、CPBL / MLB / 3D 予測モジュールが利用できます",
        "tab_login": "ログイン",
        "tab_register": "新規登録",
        "username": "ユーザー名",
        "password": "パスワード",
        "confirm_password": "パスワード（確認）",
        "btn_login": "ログイン",
        "btn_register": "登録",
        "err_invalid": "⚠️ ユーザー名またはパスワードが正しくありません。",
        "err_empty": "⚠️ ユーザー名とパスワードは必須です。",
        "err_exists": "⚠️ このユーザー名はすでに使用されています。別の名前を選んでください。",
        "err_mismatch": "⚠️ パスワードが一致しません。",
        "success_login": "✅ ログインしました！",
        "success_register": "✅ 登録が完了しました！「ログイン」タブから新しいアカウントでログインしてください。",
    },
}

l = ui_kit.language_switcher()
def t(key): return LANG[l].get(key, key)

ui_kit.hero_banner(t("title"), t("subtitle"), icon="🔐")

col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    with st.container(border=True):
        tab_login, tab_register = st.tabs([t("tab_login"), t("tab_register")])

        with tab_login:
            with st.form("login_form"):
                login_username = st.text_input(t("username"), key="login_username")
                login_password = st.text_input(t("password"), type="password", key="login_password")
                login_submitted = st.form_submit_button(t("btn_login"), use_container_width=True, type="primary")

            if login_submitted:
                if auth.verify(login_username, login_password):
                    st.session_state["user"] = login_username.strip()
                    st.success(t("success_login"))
                    st.rerun()
                else:
                    st.error(t("err_invalid"))

        with tab_register:
            with st.form("register_form"):
                reg_username = st.text_input(t("username"), key="reg_username")
                reg_password = st.text_input(t("password"), type="password", key="reg_password")
                reg_confirm = st.text_input(t("confirm_password"), type="password", key="reg_confirm")
                reg_submitted = st.form_submit_button(t("btn_register"), use_container_width=True, type="primary")

            if reg_submitted:
                if not reg_username.strip() or not reg_password:
                    st.error(t("err_empty"))
                elif reg_password != reg_confirm:
                    st.error(t("err_mismatch"))
                else:
                    ok, reason = auth.register(reg_username, reg_password)
                    if ok:
                        st.success(t("success_register"))
                    elif reason == "exists":
                        st.error(t("err_exists"))
                    else:
                        st.error(t("err_empty"))
