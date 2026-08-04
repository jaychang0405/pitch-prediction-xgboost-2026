# -*- coding: utf-8 -*-
"""從 CPBL 官網 (cpbl.com.tw) 即時擷取球隊戰績與單項排行榜 TOP5。
純函式、不依賴 streamlit，方便在 views/cpbl_app.py 用 st.cache_data 包一層。
"""
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.cpbl.com.tw"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.cpbl.com.tw/",
}
REQUEST_TIMEOUT = 15

# 單項排行榜前 5 個 item 是投手數據，後 5 個是打者數據
TOPLIST_PITCHING_COUNT = 5


def fetch_toplist():
    """回傳 (pitching_categories, batting_categories, error)。每個 category 為
    {"title_zh", "title_en", "leaders": [{"rank","name","team","value"}], "photo_url"}
    失敗時回傳 (None, None, error_message)，成功時 error 為 None。
    """
    try:
        resp = requests.get(f"{BASE_URL}/stats/toplist", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        categories = []
        for item in soup.select("div.TopFiveList > div.item"):
            title_el = item.select_one(".title")
            if not title_el:
                continue
            en_el = title_el.select_one(".en")
            title_en = en_el.get_text(strip=True) if en_el else ""
            title_zh = title_el.get_text(strip=True).replace(title_en, "")

            leaders = []
            for li in item.select("ul li"):
                rank_el = li.select_one(".rank")
                name_el = li.select_one(".player .name")
                team_el = li.select_one(".player .team")
                num_el = li.select_one(".num")
                leaders.append({
                    "rank": rank_el.get_text(strip=True) if rank_el else "",
                    "name": name_el.get_text(strip=True) if name_el else "",
                    "team": (team_el.get_text(strip=True) if team_el else "").strip("()"),
                    "value": num_el.get_text(strip=True) if num_el else "",
                })

            photo_url = None
            photo_el = item.select_one(".photo_player_1st a")
            if photo_el and photo_el.get("style") and "url(" in photo_el["style"]:
                raw = photo_el["style"].split("url(")[-1].split(")")[0].strip("'\"")
                photo_url = f"{BASE_URL}{raw}" if raw.startswith("/") else raw

            if leaders:
                categories.append({
                    "title_zh": title_zh,
                    "title_en": title_en,
                    "leaders": leaders,
                    "photo_url": photo_url,
                })

        if not categories:
            return None, None, "頁面結構解析失敗：找不到任何排行榜項目"
        return categories[:TOPLIST_PITCHING_COUNT], categories[TOPLIST_PITCHING_COUNT:], None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def fetch_standings():
    """回傳 (teams, error)。teams 每隊為 {rank, team, logo_url, games, record,
    win_pct, games_behind, home_record, away_record, streak, last10}。
    失敗時回傳 (None, error_message)，成功時 error 為 None。
    """
    try:
        resp = requests.get(f"{BASE_URL}/standings/season", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        logo_map = {}
        for a in soup.select(".teams li a"):
            name = a.get("title", "").strip()
            style = a.get("style", "")
            if name and "url(" in style:
                raw = style.split("url(")[-1].split(")")[0].strip("'\"")
                logo_map[name] = raw if raw.startswith("http") else f"{BASE_URL}{raw}"

        first_table = soup.find("table")
        if first_table is None:
            return None, "頁面結構解析失敗：找不到戰績表格"

        teams = []
        for tr in first_table.find_all("tr"):
            rank_el = tr.select_one(".rank")
            if not rank_el or not rank_el.get_text(strip=True).isdigit():
                continue
            team_el = tr.select_one(".team-w-trophy a, .sticky_wrap a")
            team_name = team_el.get_text(strip=True) if team_el else ""
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) < 16:
                continue
            teams.append({
                "rank": rank_el.get_text(strip=True),
                "team": team_name,
                "logo_url": logo_map.get(team_name),
                "games": tds[1],
                "record": tds[2],
                "win_pct": tds[3],
                "games_behind": tds[4],
                "home_record": tds[12],
                "away_record": tds[13],
                "streak": tds[14],
                "last10": tds[15],
            })

        if not teams:
            return None, "頁面結構解析失敗：找不到任何球隊列"
        return teams, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
