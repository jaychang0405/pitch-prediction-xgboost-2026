# -*- coding: utf-8 -*-
"""從野球革命 (rebas.tw) 的公開 JSON API 取得 CPBL 球隊戰績與單項排行榜 TOP5。
純函式、不依賴 streamlit，方便在 views/cpbl_app.py 用 st.cache_data 包一層。

改用 rebas.tw 而非 cpbl.com.tw 官網：官網頁面是伺服端組好的 HTML，但重新導向/
Cookie 行為在本機與 Streamlit Cloud 之間不一致（本機正常、雲端回 404），且沒有
穩定 API。rebas.tw 有清楚的 JSON API，資料經與官網比對數字一致。
"""
import requests

BASE_URL = "https://www.rebas.tw"
LEAGUE_UNIQID = "CPBL"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
REQUEST_TIMEOUT = 12
DEFAULT_TEAM_COLOR = "#42a5f5"


def _get_json(path, params=None):
    resp = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        raise RuntimeError(payload.get("message") or payload.get("error_code") or "unknown API error")
    return payload["data"]


def _current_season_uniqid():
    """球季代碼 (如 CPBL-2026-oB) 不是固定規則，每年都不同，所以動態從聯盟資料抓
    目前最新的例行賽球季，而不是寫死年份。"""
    league = _get_json(f"/api/leagues/{LEAGUE_UNIQID}")
    regular_seasons = [s for s in league.get("seasons", []) if s.get("type") == "LEAGUE_MATCHES"]
    if not regular_seasons:
        raise RuntimeError("找不到例行賽球季資料")
    regular_seasons.sort(key=lambda s: s.get("year", 0), reverse=True)
    return regular_seasons[0]["uniqid"]


def _team_color(team):
    hex_color = team.get("hex_color") or ""
    return hex_color.split(";")[0] or DEFAULT_TEAM_COLOR


# rebas.tw 用 section 這個參數同時表達「全年度/上半季/下半季」(team 類型) 和
# 「基礎/進階數據」(player 類型)，兩者字面上剛好都叫 section 但意思不同。
# 這裡只給 team 戰績用，"standard"=全年度、"top"=上半季、"bottom"=下半季。
STANDINGS_RANGES = ("standard", "top", "bottom")


def fetch_standings(season_range="standard"):
    """回傳 (teams, error)。season_range 為 "standard"(全年度) / "top"(上半季) /
    "bottom"(下半季)。teams 每隊為 {rank, team, hex_color, games, record,
    win_pct, games_behind, streak}。失敗時回傳 (None, error_message)。
    """
    if season_range not in STANDINGS_RANGES:
        season_range = "standard"
    try:
        season_id = _current_season_uniqid()
        rows = _get_json(f"/api/seasons/{season_id}/leaders", params={"type": "team", "section": season_range})

        teams = []
        for row in rows:
            team = row.get("team", {})
            wins, loses, draws = row.get("wins", 0), row.get("loses", 0), row.get("draws", 0)
            streak_n = row.get("STRK", 0)
            if streak_n > 0:
                streak = f"{streak_n}W"
            elif streak_n < 0:
                streak = f"{abs(streak_n)}L"
            else:
                streak = "-"
            gb = row.get("GB", 0)
            teams.append({
                "rank": row.get("order"),
                "team": team.get("name", ""),
                "hex_color": _team_color(team),
                "games": wins + loses + draws,
                "record": f"{wins}-{draws}-{loses}",
                "win_pct": f"{row.get('PCT', 0):.3f}",
                "games_behind": "-" if not gb else f"{gb:.1f}",
                "streak": streak,
            })

        if not teams:
            return None, "找不到任何球隊資料"
        return teams, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# (顯示名稱, JSON 欄位, 是否需先過濾 reach_min 才有意義的比較基準)
# ERA/AVG 是比率數據，樣本太小會失真，所以只看有達到「規定局數/打席」門檻的球員；
# 勝投/救援/中繼/三振/安打/全壘打/打點/盜壘是累計數據，賽季累積出來的自然就有意義，
# 而且官方規定門檻是用「局數」算的，中繼/救援投手通常局數不夠、reach_min 會是
# False，套用該門檻反而會把所有終結者都濾掉。
PITCHING_CATEGORIES = [
    ("防禦率", "ERA", "ERA", True, False),
    ("勝投", "W", "R_W", False, True),
    ("救援成功", "SV", "R_SV", False, True),
    ("中繼成功", "HLD", "R_H", False, True),
    ("奪三振", "SO", "SO", False, True),
]
BATTING_CATEGORIES = [
    ("打擊率", "AVG", "AVG", True, True),
    ("安打數", "H", "H", False, True),
    ("全壘打", "HR", "HR", False, True),
    ("打點", "RBI", "RBI", False, True),
    ("盜壘成功", "SB", "SB", False, True),
]


def _format_value(value, field):
    if field == "AVG" or field == "ERA":
        return f"{value:.3f}" if field == "AVG" else f"{value:.2f}"
    return str(value)


def _team_color_map(season_id):
    rows = _get_json(f"/api/seasons/{season_id}/leaders", params={"type": "team", "section": "standard"})
    return {row["team"]["abbr"]: _team_color(row["team"]) for row in rows if row.get("team", {}).get("abbr")}


def _build_categories(players, categories, color_map):
    result = []
    for title_zh, title_en, field, needs_qualifier, reverse in categories:
        pool = [p for p in players if p.get(field) is not None]
        if needs_qualifier:
            pool = [p for p in pool if p.get("reach_min")]
        pool.sort(key=lambda p: p.get(field, 0), reverse=reverse)
        leaders = []
        for i, p in enumerate(pool[:5], start=1):
            player = p.get("player", {})
            team_abbr = player.get("team_abbr", "")
            leaders.append({
                "rank": str(i),
                "name": player.get("name", ""),
                "team": team_abbr,
                "value": _format_value(p.get(field), field),
                "hex_color": color_map.get(team_abbr, DEFAULT_TEAM_COLOR),
            })
        if leaders:
            result.append({"title_zh": title_zh, "title_en": title_en, "leaders": leaders})
    return result


def fetch_toplist():
    """回傳 (pitching_categories, batting_categories, error)。每個 category 為
    {"title_zh", "title_en", "leaders": [{"rank","name","team","value","hex_color"}]}。
    失敗時回傳 (None, None, error_message)。
    """
    try:
        season_id = _current_season_uniqid()
        color_map = _team_color_map(season_id)
        pitchers = _get_json(f"/api/seasons/{season_id}/leaders", params={"type": "pitcher", "section": "standard"})
        batters = _get_json(f"/api/seasons/{season_id}/leaders", params={"type": "batter", "section": "standard"})

        pitching = _build_categories(pitchers, PITCHING_CATEGORIES, color_map)
        batting = _build_categories(batters, BATTING_CATEGORIES, color_map)

        if not pitching or not batting:
            return None, None, "排行榜資料為空"
        return pitching, batting, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
