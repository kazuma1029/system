# createDataset.py
# -*- coding: utf-8 -*-
import re
import time
from typing import Optional, List
import requests
from bs4 import BeautifulSoup, Tag, NavigableString
import pandas as pd

MOVIE_ID = "148"  # ジュマンジ
BASE_CGI = f"https://www.jtnews.jp/cgi-bin/review.cgi?TITLE_NO={MOVIE_ID}"
BASE_NEW = f"https://www.jtnews.jp/review/{MOVIE_ID}/"
OUTPUT_XLSX = "jumanji_reviews.xlsx"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Referer": "https://www.jtnews.jp/index.html",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
    "Cache-Control": "no-cache",
}
SLEEP_SEC = 1.2
MAX_PAGES = 50

RE_REV_DIV_ID   = re.compile(r"^REV_\d+$")
RE_LEADING_NUM  = re.compile(r"^\s*\d+．\s*")
RE_SCORE_TEXT   = re.compile(r"(\d{1,2})\s*点")
RE_REVIEWER_HREF= re.compile(r"(revper|REVPER_NO|/reviewer|/user|PERSONAL)", re.I)

session = requests.Session()
session.headers.update(HEADERS)

def get_soup(url: str) -> Optional[BeautifulSoup]:
    try:
        r = session.get(url, timeout=20, allow_redirects=True)
        if r.status_code == 403:
            time.sleep(2.0)
            r = session.get(url, timeout=20, allow_redirects=True)
        if r.status_code != 200:
            print(f"[WARN] HTTP {r.status_code} for {url}")
            return None
        return BeautifulSoup(r.text, "lxml")
    except requests.RequestException as e:
        print(f"[ERROR] request failed: {e}")
        return None

def extract_review_text(div: Tag) -> Optional[str]:
    for b in div.select("span.badge"):  # ネタバレ等のバッジ除去
        b.decompose()
    raw = " ".join(div.stripped_strings)
    if not raw:
        return None
    raw = RE_LEADING_NUM.sub("", raw)   # 先頭の「241．」等を除去
    text = re.sub(r"\s+", " ", raw).strip()
    return text if len(text) >= 10 else None

def find_score_near(div: Tag) -> Optional[int]:
    parent = div.parent
    if isinstance(parent, Tag):
        for sp in parent.find_all("span"):
            m = RE_SCORE_TEXT.search(sp.get_text(" ", strip=True))
            if m:
                return int(m.group(1))
    sib = div.previous_sibling
    while isinstance(sib, (Tag, NavigableString)):
        if isinstance(sib, Tag):
            m = RE_SCORE_TEXT.search(sib.get_text(" ", strip=True))
            if m:
                return int(m.group(1))
        sib = sib.previous_sibling
    sib = div.next_sibling
    while isinstance(sib, (Tag, NavigableString)):
        if isinstance(sib, Tag):
            m = RE_SCORE_TEXT.search(sib.get_text(" ", strip=True))
            if m:
                return int(m.group(1))
        sib = sib.next_sibling
    root = div
    while root.parent and isinstance(root.parent, Tag):
        root = root.parent
    for sp in root.find_all("span"):
        m = RE_SCORE_TEXT.search(sp.get_text(" ", strip=True))
        if m:
            return int(m.group(1))
    return None

def find_reviewer_near(div: Tag) -> Optional[str]:
    parent = div.parent
    if isinstance(parent, Tag):
        a = parent.find("a", href=RE_REVIEWER_HREF)
        if a:
            name = a.get_text(" ", strip=True)
            if name:
                return name
    node = div
    while node:
        a = node.find("a", href=RE_REVIEWER_HREF)
        if a:
            name = a.get_text(" ", strip=True)
            if name:
                return name
        node = node.parent if isinstance(node.parent, Tag) else None
    root = div
    while root.parent and isinstance(root.parent, Tag):
        root = root.parent
    for a in root.find_all("a", href=RE_REVIEWER_HREF):
        name = a.get_text(" ", strip=True)
        if name:
            return name
    return None

def parse_reviews_from_soup(soup: BeautifulSoup) -> List[dict]:
    rows = []
    for div in soup.find_all("div", id=RE_REV_DIV_ID):
        body = extract_review_text(div)
        if not body:
            continue
        score = find_score_near(div)
        reviewer = find_reviewer_near(div) or ""
        rows.append({
            "reviewer": reviewer,
            "score": score,
            "review_text": body
        })
    return rows

def crawl_all_pages_cgi() -> List[dict]:
    all_rows: List[dict] = []
    for p in range(1, MAX_PAGES + 1):
        url = BASE_CGI if p == 1 else f"{BASE_CGI}&PAGE_NO={p}"
        print(f"[INFO] Fetch: {url}")
        soup = get_soup(url)
        if not soup:
            break
        rows = parse_reviews_from_soup(soup)
        if not rows:
            if p == 1 and not all_rows:
                break
            else:
                break
        all_rows.extend(rows)
        time.sleep(SLEEP_SEC)
    return all_rows

def crawl_new_design_once() -> List[dict]:
    print(f"[INFO] Fallback: {BASE_NEW}")
    soup = get_soup(BASE_NEW)
    if not soup:
        return []
    return parse_reviews_from_soup(soup)

def main():
    rows = crawl_all_pages_cgi()
    if not rows:
        rows = crawl_new_design_once()

    if not rows:
        print("[ERROR] レビューを抽出できませんでした。サイト構造の変化の可能性があります。")
        return

    df = pd.DataFrame(rows, columns=["reviewer", "score", "review_text"])
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"[DONE] {len(df)} rows -> {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
