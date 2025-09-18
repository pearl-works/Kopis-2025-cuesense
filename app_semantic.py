# app_semantic.py — 공연추천 챗봇 
# -----------------------------------------------------------
# 2025년 KOPIS 공모전 서비스개발부문
# 가보자KU 전재현, 서진주
# - Cuesense
# -----------------------------------------------------------

import os
import re
import math
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import base64
from pathlib import Path

import pandas as pd
import numpy as np
import pytz
import streamlit as st
from dotenv import load_dotenv
from html import escape as html_escape

from google.cloud import bigquery
from google.oauth2 import service_account
from rank_bm25 import BM25Okapi

from openai import OpenAI

# --------------------------
# 0) 환경설정
# --------------------------
load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# PROJECT_ID = os.getenv("PROJECT_ID")
# SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")

#Secrets 읽기
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "").strip()
PROJECT_ID = st.secrets.get("PROJECT_ID", "").strip()
GCP_SA_INFO = st.secrets.get("gcp_service_account", None)


assert OPENAI_API_KEY.strip(), "환경변수 OPENAI_API_KEY를 먼저 설정하세요."
if not PROJECT_ID:
    raise RuntimeError("PROJECT_ID가 설정되지 않았어요 (.env 또는 환경변수 확인).")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# def get_bq_client(project_id: Optional[str] = None,
#                   service_account_file: Optional[str] = None) -> bigquery.Client:
#     if service_account_file:
#         creds = service_account.Credentials.from_service_account_file(service_account_file)
#         return bigquery.Client(project=project_id or creds.project_id, credentials=creds)
#     return bigquery.Client(project=project_id)

# bq_client = get_bq_client(PROJECT_ID, SERVICE_ACCOUNT_FILE)

def get_bq_client(project_id: str, sa_info: dict) -> bigquery.Client:
    if not sa_info:
        st.error("gcp_service_account가 Secrets에 없습니다.")
        st.stop()
    creds = service_account.Credentials.from_service_account_info(sa_info)
    return bigquery.Client(project=project_id or sa_info.get("project_id"), credentials=creds)

bq_client = get_bq_client(PROJECT_ID, GCP_SA_INFO)

KST = pytz.timezone("Asia/Seoul")
TODAY = dt.datetime.now(KST)

# st.set_page_config(page_title="CueSence 큐센스", page_icon="🎭", layout="wide")
with open("assets/cuesence_favicon.png", "rb") as f:
    st.set_page_config(page_title="CueSence 큐센스", page_icon=f.read(), layout="wide")


# --------------------------
# 공통 유틸
# --------------------------
def fn_query(sql: str) -> pd.DataFrame:
    return bq_client.query(sql).result().to_dataframe()

def tok(s: str) -> List[str]:
    return re.findall(r"[ㄱ-ㅎ가-힣A-Za-z0-9]+", (s or ""))

def is_gpt5(model: str) -> bool:
    return model.lower().startswith("gpt-5-mini")

def chat_complete(model: str, messages: list, temperature: Optional[float] = None, **kwargs):
    payload = {"model": model, "messages": messages}
    if (temperature is not None) and (not is_gpt5(model)):
        payload["temperature"] = float(temperature)
    payload.update(kwargs or {})
    return openai_client.chat.completions.create(**payload)

# --------------------------
# 1) 마스터 데이터
# --------------------------
@st.cache_data(show_spinner=False)
def load_master_lists():
    try:
        df_genres = fn_query("SELECT genre_name FROM pg_snapshot.dm_perform_genre")
        df_cities = fn_query("SELECT REPLACE(region_name,' ','') AS region_name FROM pg_snapshot.dm_facility_region")
        df_venues = fn_query("SELECT REPLACE(facility_name,' ','') AS facility_name FROM pg_snapshot.dm_facility_name")
        return df_genres["genre_name"].tolist(), df_cities["region_name"].tolist(), df_venues["facility_name"].tolist()
    except Exception:
        # 마스터 테이블 없으면 빈값으로
        return [], [], []

GENRES, CITIES, VENUES = load_master_lists()

# === 주소 가제티어(서브로케이션 후보) 만들기 ===
_GEO_SUFFIX = r"(구|군|동|로|길|역|가|읍|면|리)$"
_SPECIAL_NAMES = r"(대학로|홍대|성수|잠실|압구정|청담|광화문|여의도|상암|강남|서초|종로|명동|이태원|한남|신촌|합정|왕십리|건대|건대입구|잠원|용산|마포|상암|성수|선릉|삼성|역삼|사당|교대|노원|수유|상봉|중랑|구로|신도림|목동|망원|합정|성북|동대문|을지로|충무로|신사)"

def build_address_gazetteer(df: pd.DataFrame, max_len:int=6) -> List[str]:
    gaz = set()
    if "address" not in df.columns or df.empty:
        return []
    for addr in df["address"].dropna().astype(str):
        for tok in re.split(r"[ ,/()\-·]+", addr):
            tok = tok.strip()
            if not tok or len(tok) < 2: 
                continue
            if re.search(_GEO_SUFFIX, tok) or re.search(_SPECIAL_NAMES, tok):
                if tok not in {"대한민국", "서울특별시", "서울시"}:
                    gaz.add(tok.replace("서울특별시","서울").replace("서울시","서울"))
    gaz = {g for g in gaz if 2 <= len(g) <= max_len}
    # 너무 많은 경우 상위 N만 사용(길이 긴 고유명 우선)
    return sorted(gaz, key=lambda x: (-len(x), x))[:600]  # LLM 프롬프트 길이 보호


def extract_sublocs_llm(query: str, gazetteer: List[str], top_k:int=6) -> List[str]:
    """
    질의문과 주소 가제티어를 주면, LLM이 질의에 "강하게 함의"된 지명(강남/대학로/성수 등)만 추려서 반환.
    - 반환 형식: JSON array of strings (예: ["강남","대학로"])
    - gazetteer 바깥 단어는 버림 → 과적합/환각 방지
    """
    if not query or not gazetteer:
        return []
    sys = (
        "너는 공연 검색 보조자다. 사용자의 문장을 읽고, 문장 안에서 암시되는 "
        "세부 지역 키워드(예: 강남, 대학로, 성수, 홍대 등)만 '가제티어 목록'에서 골라 JSON 배열로만 반환해라. "
        "가제티어에 없는 단어는 절대 포함하지 마라. 일반 단어(예: 이번, 추천해줘, 바이올린)는 포함 금지. "
        "최대 6개. 출력은 반드시 JSON 배열 하나만."
    )
    gaz_str = " / ".join(gazetteer)
    usr = (
        f"질문: {query}\n\n"
        f"[가제티어(후보 지명)]\n{gaz_str}\n\n"
        "출력 예시: [\"강남\",\"대학로\"]"
    )
    try:
        resp = chat_complete(
            CHAT_MODEL,
            [{"role":"system","content":sys},{"role":"user","content":usr}],
            temperature=0.0,
            response_format={"type": "json_object"} if is_gpt5(CHAT_MODEL) else None
        )
        txt = resp.choices[0].message.content.strip()
        # gpt-5-mini가 오브젝트로 줄 수 있어 후처리(배열만 남기기)
        try:
            import json
            obj = json.loads(txt)
            if isinstance(obj, list):
                arr = obj
            elif isinstance(obj, dict):
                # {"sublocs":[...]} 또는 {"result":[...]} 류 대응
                arr = obj.get("sublocs") or obj.get("result") or obj.get("locations") or []
            else:
                arr = []
            # gazetteer 교집합 + 길이 제한
            arr = [s for s in arr if isinstance(s, str)]
        except Exception:
            # 혹시 배열 문자열로 바로 온 경우
            m = re.search(r"\[.*\]", txt, re.S)
            import json
            arr = json.loads(m.group(0)) if m else []
        # 안전필터: 가제티어 밖 단어 제거 + 중복 제거 + 상한
        sset = []
        gset = set(gazetteer)
        for s in arr:
            s = s.strip()
            if s in gset and s not in sset and 2 <= len(s) <= 10:
                sset.append(s)
            if len(sset) >= top_k:
                break
        return sset
    except Exception:
        return []


# --------------------------
# 2) 쿼리 파서 (기간/지역/장르/장소/예산/동반자)
# --------------------------
@dataclass
class QueryFilters:
    start_date: dt.datetime
    end_date: dt.datetime
    city: Optional[str]
    genre: Optional[str]
    venue: Optional[str]
    max_price: Optional[int]
    with_companion: Optional[str]
    sublocs: Optional[List[str]] = None

def to_kst_midnight(ts: dt.datetime) -> dt.datetime:
    if getattr(ts, "tzinfo", None) is None:
        return KST.localize(ts.replace(hour=0,minute=0,second=0,microsecond=0))
    return ts.astimezone(KST).replace(hour=0,minute=0,second=0,microsecond=0)

def parse_user_query(q: str, base: dt.datetime, subloc_gaz: Optional[List[str]] = None) -> QueryFilters:
    q = (q or "").strip()
    base = to_kst_midnight(base)

    # 기본: 앞으로 30일
    start = base
    end   = start + dt.timedelta(days=30)

    q_norm = q.replace(" ", "")

    # === 기간 파싱 (생략: 기존 그대로) ===
    if "이번주" in q_norm and "주말" not in q_norm and "다음주" not in q_norm:
        week_start = start - dt.timedelta(days=start.weekday())
        start = week_start
        end   = week_start + dt.timedelta(days=7)
    if "이번주말" in q_norm or "이번 주말" in q:
        weekday = start.weekday()
        start = start + dt.timedelta(days=(5 - weekday) % 7)
        end   = start + dt.timedelta(days=2)
    if "다음주" in q_norm:
        weekday = start.weekday()
        start = start + dt.timedelta(days=(7 - weekday))
        end   = start + dt.timedelta(days=7)
    if "오늘" in q or "금일" in q:
        end = start + dt.timedelta(days=1)

    genre = next((g for g in GENRES  if g and g in q), None)
    city  = next((c for c in CITIES  if c and c in q), None)
    venue = next((v for v in VENUES  if v and v in q), None)

    max_price = 50000 if any(k in q for k in ["저렴","가성비","5만","5만원","50,000","50000"]) else None
    companion = "커플" if any(k in q for k in ["여자친구","남자친구","커플","데이트"]) else \
                "가족" if any(k in q for k in ["아이","가족","부모님"]) else \
                "친구" if any(k in q for k in ["친구","동료"]) else None

    sublocs = extract_sublocs_llm(q, subloc_gaz or []) if subloc_gaz else None

    return QueryFilters(start, end, city, genre, venue, max_price, companion, sublocs)


def detect_intent(q: str) -> str:
    if not q: 
        return "unknown"
    qs = q.strip().lower()
    # 이름 질문
    if any(k in qs for k in ["이름", "who are you", "what is your name", "너 이름", '안녕']):
        return "ask_name"
    # 공연 추천/검색 의도 키워드
    rec_kw = ["추천", "찾아줘", "공연", "페스티벌", "클래식", "연극", "뮤지컬", "콘서트", "오페라", "재즈", "알려줘"]
    if any(k in q for k in rec_kw):
        return "recommend"
    return "other"


# --------------------------
# 3) SHOW 로드 
# --------------------------
@st.cache_data(show_spinner=True)

def load_show_candidates(day_setting) -> pd.DataFrame:
    sql = (f"""
            WITH clean AS (
            SELECT
                CAST(performance_code AS STRING)                   AS performance_code,
                CAST(performance_name AS STRING)                   AS performance_name,
                CAST(venue_code AS STRING)                         AS venue_code,
                CAST(venue_name AS STRING)                         AS venue_name,
                CAST(region_name AS STRING)                        AS region_name,
                CAST(address AS STRING)                        	   AS address,
                CAST(genre_name AS STRING)                         AS genre_name,
                CAST(subgenre_name AS STRING)                      AS subgenre_name,
                CAST(production_company_name AS STRING)            AS production_company_name,
                CAST(cast_info AS STRING)                          AS cast_info,
                PARSE_TIMESTAMP('%Y-%m-%d %H:%M', performance_ts, 'Asia/Seoul')    AS performance_ts,
                PARSE_TIMESTAMP('%Y-%m-%d %H:%M', booking_cancel_ts, 'Asia/Seoul') AS booking_cancel_ts,
                cast(replace(seats_for_sale, '석','') AS int) AS seats_for_sale ,
                CAST(CAST(unit_price AS NUMERIC) AS INT64)                  AS unit_price,
                CAST(booking_cancel_type AS STRING)                AS booking_cancel_type,
                CAST(CAST(booking_cancel_qty AS NUMERIC) AS INT64) AS booking_cancel_qty
            FROM pg_snapshot.dw_kopis_row 
            ),
            base AS (
            -- "미래 회차" 목록 (중복 제거)
            SELECT
                performance_code,
                ANY_VALUE(performance_name)           AS performance_name,
                ANY_VALUE(venue_code)                 AS venue_code,
                ANY_VALUE(venue_name)                 AS venue_name,
                ANY_VALUE(region_name)                AS region_name,
                ANY_VALUE(address)                	  AS address,
                ANY_VALUE(genre_name)                 AS genre_name,
                ANY_VALUE(subgenre_name)              AS subgenre_name,
                ANY_VALUE(production_company_name)    AS production_company_name,
                ANY_VALUE(cast_info)                  AS cast_info,
                performance_ts,
                ANY_VALUE(seats_for_sale)             AS seats_for_sale,
                ANY_VALUE(unit_price)                 AS unit_price
            FROM clean
            WHERE performance_ts >= CAST('{day_setting}' AS TIMESTAMP)
            GROUP BY performance_code, performance_ts
            ),
            recent_30 AS (
            -- 최근 30일 순판매량(예매-취소), 공연코드 단위
            SELECT
                performance_code,
                SUM(CASE WHEN booking_cancel_type = '2'
                        THEN -COALESCE(booking_cancel_qty,0)
                        ELSE  COALESCE(booking_cancel_qty,0)
                    END) AS net_qty_30d
            FROM clean
            WHERE booking_cancel_ts >= TIMESTAMP_SUB(CAST('{day_setting}' AS TIMESTAMP), INTERVAL 30 DAY)
            GROUP BY performance_code
            ),
            genre_minmax AS (
            -- 장르별 min/max로 정규화 준비
            SELECT
                c.performance_code,
                c.genre_name,
                MIN(r.net_qty_30d) OVER (PARTITION BY c.genre_name) AS min_g,
                MAX(r.net_qty_30d) OVER (PARTITION BY c.genre_name) AS max_g,
                r.net_qty_30d
            FROM (SELECT DISTINCT performance_code, genre_name FROM clean) c
            LEFT JOIN recent_30 r USING (performance_code)
            ),
            popularity_scaled AS (
            -- 0~1 스케일의 popularity
            SELECT
                performance_code,
                CASE
                WHEN max_g IS NULL OR min_g IS NULL OR max_g = min_g THEN 0.0
                ELSE SAFE_DIVIDE(net_qty_30d - min_g, max_g - min_g)
                END AS popularity
            FROM genre_minmax
            ),
            sold_until_now AS (
            -- as_of_ts 시점까지 회차별 누적 순판매
            SELECT
                performance_code,
                performance_ts,
                SUM(CASE WHEN booking_cancel_type = '2'
                        THEN -COALESCE(booking_cancel_qty,0)
                        ELSE  COALESCE(booking_cancel_qty,0)
                    END) AS net_sold
            FROM clean
            WHERE booking_cancel_ts <= CAST('{day_setting}' AS TIMESTAMP)
            GROUP BY performance_code, performance_ts
            )
            SELECT
                CONCAT(
                        b.performance_code, '_',
                        FORMAT_TIMESTAMP('%Y%m%d%H%M', b.performance_ts, 'Asia/Seoul')
                    ) AS show_id,
                b.performance_code, 
                b.performance_ts, 
                b.performance_name  AS title,
                b.genre_name        AS genre,
                b.subgenre_name     AS subgenre,
                b.venue_name        AS venue,
                b.region_name       AS city,
                b.address           AS address,
                b.performance_ts    AS datetime,
                b.unit_price        AS price_avg,
                b.production_company_name AS producer,
                IFNULL(b.cast_info, '') AS cast_text,            -- ← cast 예약어 혼동 방지
                IFNULL(ps.popularity, 0.0) AS popularity,        -- 0~1 (낮을수록 비인기)
                CASE WHEN COALESCE(b.seats_for_sale,0) >= 1000 THEN 1 ELSE 0 END AS is_major,
                COALESCE(b.seats_for_sale, 0) AS seats_total,
                GREATEST(0, COALESCE(b.seats_for_sale,0) - COALESCE(s.net_sold,0)) AS seats_left,
                acs.accessible_seat_count as accessible_seat_count,
                url.sales_page_url as sales_page_url,
                openai.opnai_output
                FROM base b
                    LEFT JOIN popularity_scaled ps 
                        ON b.performance_code = ps.performance_code
                    LEFT JOIN sold_until_now s 
                         ON b.performance_code = s.performance_code 
                        AND b.performance_ts = s.performance_ts
                    INNER JOIN pg_snapshot.dm_openai_output openai
                         ON b.performance_code = openai.performance_code
                    LEFT JOIN  pg_snapshot.dm_accessible_seat acs
                         ON b.venue_code = acs.venue_code
                    LEFT JOIN pg_snapshot.dm_performance_interpark url
                         ON b.performance_code = url.performance_code
                         AND b.performance_ts  = PARSE_TIMESTAMP('%Y-%m-%d %H:%M', url.performance_ts, 'Asia/Seoul')
                    ORDER BY b.performance_ts
            """
        )

    sql = ('''SELECT *
          FROM pg_snapshot.temp_df_show3
          ORDER BY 1''')

    df = fn_query(sql)
    # datetime 표준화(KST 표시용)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_convert("Asia/Seoul")
    # 결측 보정
    for c in ["title","genre","venue","city","price_avg","opnai_output"]:
        if c in df.columns:
            df[c] = df[c].fillna("")

    return df

# --------------------------
# 4) 필터 적용 → 후보 축소
# --------------------------
def apply_filters(df: pd.DataFrame, f: QueryFilters) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()

    # 기간
    if "datetime" in out.columns:
        out = out[(out["datetime"] >= f.start_date) & (out["datetime"] < f.end_date)]

    # 지역/장르/장소
    if f.city and "city" in out.columns:
        out = out[out["city"].astype(str).str.contains(re.escape(f.city), na=False)]
    if f.genre and "genre" in out.columns:
        out = out[out["genre"].astype(str).str.contains(re.escape(f.genre), na=False)]
    if f.venue and "venue" in out.columns:
        out = out[out["venue"].astype(str).str.contains(re.escape(f.venue), na=False)]

    # 서브로케이션: address에 부분일치 (여러 개면 OR)
    if getattr(f, "sublocs", None) and "address" in out.columns:
        pat = "|".join(re.escape(s) for s in f.sublocs if s)
        if pat:
            out = out[out["address"].astype(str).str.contains(pat, na=False)]

    # 예산
    if f.max_price and "price_avg" in out.columns:
        with np.errstate(invalid="ignore"):
            out = out[(out["price_avg"].fillna(0) <= f.max_price)]

    # 베리어프리
    if st.session_state.get("bf_only", True):
        out = out[out["accessible_seat_count"].fillna(0) > 0]

    return out.reset_index(drop=True)


# --------------------------
# 5) 시맨틱 매칭 (BM25 → 임베딩 재랭킹)  ※ 필터링된 집합에만 수행
# --------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-5-mini")  

def compose_doc_text(row: pd.Series) -> str:
    return (
        f"공연명: {row.get('title','')}\n"
        f"공연장: {row.get('venue','')} ({row.get('city','')})\n"
        f"장르: {row.get('genre','')}\n"
        f"메타: {row.get('opnai_output','')}"
    )

def build_bm25(texts: List[str]) -> BM25Okapi:
    tokens = [tok(t.lower()) for t in texts]
    return BM25Okapi(tokens)

def embed_vec(texts: List[str]) -> np.ndarray:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts, encoding_format="float")
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def hyde(query: str, enable: bool, max_chars: int = 800) -> str:
    if not enable:
        return query
    sys = "질문을 바탕으로 한국어로 그럴듯한 가상요약을 7~9문장으로 작성."
    usr = f"질문: {query}\n- 공연명/장르/분위기/감정/장소 키워드를 포함"
    resp = chat_complete(CHAT_MODEL, [{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.3)
    return (resp.choices[0].message.content or "")[:max_chars]

def semantic_match(query: str, df_filt: pd.DataFrame, n_bm25: int, top_k: int, use_hyde: bool):
    if df_filt.empty:
        return [], {}, pd.DataFrame()
    docs = df_filt.apply(compose_doc_text, axis=1).tolist()
    bm25 = build_bm25(docs)
    q_tokens = [t.lower() for t in tok(query)]
    scores = bm25.get_scores(q_tokens)
    order = np.argsort(-np.asarray(scores))[:max(n_bm25, top_k)]
    cand = df_filt.iloc[order].reset_index(drop=True)

    # 임베딩 재랭킹
    q_text = hyde(query, enable=use_hyde)
    q_vec = embed_vec([q_text])[0]; q_vec /= (np.linalg.norm(q_vec)+1e-9)
    c_mat = embed_vec(cand.apply(compose_doc_text, axis=1).tolist())
    c_mat /= (np.linalg.norm(c_mat, axis=1, keepdims=True)+1e-9)
    sims = c_mat @ q_vec

    # 전체 후보 score map (performance_code -> sim)
    score_map = {}
    for i in range(len(cand)):
        code = str(cand.iloc[i].get("performance_code",""))
        score_map[code] = float(sims[i])

    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for i in top_idx:
        r = cand.iloc[i]
        results.append({
            "performance_code": str(r.get("performance_code","")),
            "popularity": float(r.get("popularity", 0.0)),
            "score": float(sims[i]),
            "title": r.get("title",""),
            "genre": r.get("genre",""),
            "venue": r.get("venue",""),
            "city": r.get("city",""),
            "datetime": r.get("datetime",""),
            "price_avg": r.get("price_avg",""),
            "accessible_seat_count": r.get("accessible_seat_count",""),
            "opnai_output": r.get("opnai_output",""),
        })
    return results, score_map, cand


def inject_serendipity_slot(
    ranked: list,
    df_filt: pd.DataFrame,
    score_map: dict,
    k: int,
    slot_index: int = 3,               # 4번째 (0-based)
    pop_col: str = "popularity",       # 0~1 (낮을수록 비인기)
    code_col: str = "performance_code",
) -> list:
    if not ranked or df_filt.empty:
        return ranked

    top = ranked[:k].copy()
    if slot_index >= len(top):
        slot_index = max(0, len(top) // 2)

    chosen = {str(it.get(code_col, "")) for it in top if isinstance(it, dict)}

    df_tmp = df_filt.copy()
    if pop_col not in df_tmp.columns:
        return top
    df_tmp[pop_col] = df_tmp[pop_col].fillna(0.0)

    # 하위 20% 비인기
    q20 = df_tmp[pop_col].quantile(0.20)
    pool = df_tmp[(df_tmp[pop_col] <= q20) & (~df_tmp[code_col].astype(str).isin(chosen))].copy()
    if pool.empty:
        return top

    # 시맨틱 점수로 정렬 (없으면 0)
    pool["__s"] = pool[code_col].astype(str).map(score_map).fillna(0.0)
    pool = pool.sort_values(["__s", pop_col], ascending=[False, True])

    pick = pool.head(1)
    if pick.empty:
        return top

    picked_code = str(pick.iloc[0][code_col])

    # ranked에 있는 동일 코드(후보 중) 찾아 동일 포맷으로 삽입
    ser_item = next((it for it in ranked if str(it.get(code_col, "")) == picked_code), None)
    if ser_item is None:
        # 최소 필드만 구성 (렌더러가 사용하는 키 위주)
        ser_item = {
            code_col: picked_code,
            "score": float(score_map.get(picked_code, 0.0)),
            "title": pick.iloc[0].get("title", ""),
            "genre": pick.iloc[0].get("genre", ""),
            "venue": pick.iloc[0].get("venue", ""),
            "city":  pick.iloc[0].get("city", ""),
            "datetime": pick.iloc[0].get("datetime", ""),
            "price_avg": pick.iloc[0].get("price_avg", ""),
            "accessible_seat_count": pick.iloc[0].get("accessible_seat_count", ""),
            "opnai_output": pick.iloc[0].get("opnai_output", ""),
        }

    top.insert(slot_index, ser_item)
    return top[:k]


def dedup_by_code(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df
    tmp = df.copy()
    if "datetime" in tmp.columns:
        tmp = tmp.sort_values(["performance_code", "datetime"], ascending=[True, True])
    else:
        tmp = tmp.sort_values(["performance_code"])
    return tmp.drop_duplicates(subset=["performance_code"], keep="first").reset_index(drop=True)


def dedup_ranked_by_code(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    uniq = []
    for it in items:
        code = str(it.get("performance_code",""))
        if code and code not in seen:
            uniq.append(it)
            seen.add(code)
    return uniq    

# --------------------------
# 6) 카드 템플릿 
# --------------------------
import json

def parse_meta(opnai_output: str) -> Dict[str, Any]:
    try:
        return json.loads(opnai_output) if opnai_output else {}
    except Exception:
        return {}

def render_card(item: Dict[str, Any], rank: int):

    meta = parse_meta(item.get("opnai_output",""))
    summary = meta.get("요약") or ""
    keywords = meta.get("키워드") or []
    kw = " · ".join(keywords[:8]) if isinstance(keywords, list) else ""

    # 날짜 문자열
    try:
        dt_val = item.get("datetime")
        if isinstance(dt_val, pd.Timestamp):
            dt_str = dt_val.strftime("%Y-%m-%d %a %H:%M")
        else:
            dt_str = str(dt_val or "")
    except Exception:
        dt_str = str(item.get("datetime",""))

    # 안전한 정수 포맷터
    def fmt_int(x) -> str | None:
        try:
            if x is None: return None
            if isinstance(x, float) and math.isnan(x): return None
            v = int(float(str(x).replace(",", "")))  # "1,234"도 허용
            return f"{v:,}"
        except Exception:
            return None

    price_str    = fmt_int(item.get("price_avg")) or "-"
    acc_seat_str = fmt_int(item.get("accessible_seat_count")) or "-"

    # 예매 링크 (있을 때만)
    raw_url = (item.get("sales_page_url") or "").strip()
    def normalize_url(u: str) -> str | None:
        if not u: return None
        if u.startswith(("http://","https://")): return u
        if u.startswith("www."): return "https://" + u
        return "https://" + u
    url = normalize_url(raw_url)
    link_html = (
        f"<div style='margin-top:10px;'>🎟️ <b>여기에서 예매할 수 있어요</b> — "
        f"<a href='{html_escape(url)}' target='_blank' rel='noopener noreferrer'>{html_escape(url)}</a></div>"
        if url else ""
    )

    # semantic score 툴팁(문과 톤) — 들여쓰기 없는 단일 문자열
    score_val = float(item.get("score", 0.0) or 0.0)
    score_help_html = (
        "<div style=\"font-size:12px;color:#64748b;display:flex;align-items:center;gap:6px;\">"
        f"<span>semantic score: {score_val:.4f}</span>"
        "<span style=\"cursor:help;border-bottom:1px dotted #94a3b8;\" "
        "title=\"당신의 질문과 이 공연 소개가 얼마나 ‘결이 맞는지’를 0~1로 나타낸 점수예요.\n"
        "1에 가까울수록 지금 찾는 분위기와 더 잘 어울립니다.\n"
        "(키워드가 겹치는지, 문장의 뜻이 닮았는지도 함께 살펴봐요. 같은 질문 안에서 상대적으로 맞춰 보여줍니다.)\">"
        "ⓘ help</span></div>"
    )

    # 안전 이스케이프
    title_html   = html_escape(str(item.get("title","")))
    genre_html   = html_escape(str(item.get("genre","")))
    venue_html   = html_escape(str(item.get("venue","")))
    city_html    = html_escape(str(item.get("city","")))
    dt_html      = html_escape(dt_str)
    summary_html = html_escape(summary) if summary else "이 공연이 딱 적합해 보입니다!!"
    kw_html      = html_escape(kw) if kw else ""

    st.markdown(
        f"""
<div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;margin:10px 0;background:#ffffff;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:18px;font-weight:700;">#{rank} {title_html}</div>
    {score_help_html}
  </div>

  <div style="margin-top:6px;color:#111827;display:flex;flex-wrap:wrap;gap:10px;">
    <span>🎭 <b>{genre_html}</b></span>
    <span>📍 {venue_html} ({city_html})</span>
    <span>🗓️ {dt_html}</span>
    <span>💰 {price_str}</span>
    <span>♿ 좌석 {acc_seat_str}</span>
  </div>

  <div style="margin-top:10px;line-height:1.5;">
    🧠 <b>추천 이유</b> — {summary_html}
  </div>
  {("<div style='margin-top:8px;color:#374151;'>🔑 <b>이런 느낌이에요!</b> — " + kw_html + "</div>") if kw_html else ""}
  {link_html}
</div>
""",
        unsafe_allow_html=True
    )




# --------------------------
# 7) 메인 앱
# --------------------------
def run_app():
    import base64
    from pathlib import Path

    def img_to_base64(path: str) -> str:
        p = Path(path)
        return base64.b64encode(p.read_bytes()).decode("utf-8") if p.exists() else ""

    if "chat" not in st.session_state:
        st.session_state["chat"] = []  # [{"role":"user"|"assistant","content":str,"cards":list|None}]

    # ---------------------------
    # Global CSS (one place)
    # ---------------------------
    st.markdown("""
    <style>
      /* ==== Sidebar bottom caption (sticky) ==== */
      [data-testid="stSidebar"] > div:first-child,
      section[data-testid="stSidebar"] > div {
        height: 100vh; display: flex; flex-direction: column;
      }
      .sidebar-bottom { margin-top: auto; padding: 10px 8px 16px;
                        color: rgba(49,51,63,.6); font-size: 0.8rem; }
      .sidebar-bottom hr { margin: 6px 0 8px; }

      /* ==== Hero ==== */
      .hero {
        background: linear-gradient(135deg, #fbf4f6 0%, #f4f1fb 60%, #f1eefb 100%);
        border: 1px solid rgba(0,0,0,.05);
        border-radius: 18px;
        padding: 18px 22px;
        display: flex; align-items: center; justify-content: space-between; gap: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,.06);
      }
      .hero-left { display:flex; align-items:center; gap:14px; }
      .hero-logo  { height: 64px; object-fit: contain; filter: drop-shadow(0 2px 6px rgba(0,0,0,.06)); }
      .hero-tagline { margin: 0; color:#333; opacity:.9; font-size:.95rem; }
      .pills { white-space:nowrap; }
      .pill {
        display:inline-block; padding:6px 12px; border-radius:999px;
        background: rgba(255,255,255,.75); border:1px solid rgba(0,0,0,.06);
        font-size:.85rem; color:#333; margin-left:8px; backdrop-filter: blur(4px);
      }
      @media (max-width: 900px) { .hero { flex-direction: column; align-items: flex-start; }
                                  .pills { margin-top: 8px; } }

      /* ==== Empty-state card ==== */
      .emptystate {
        border: 1px dashed rgba(0,0,0,.15);
        border-radius: 12px;
        padding: 24px 28px;
        margin: 14px 0 6px;
        text-align: center;
        background: #fafafa;
        color: rgba(60,60,67,.9);
        font-size: 0.95rem; line-height: 1.55;
      }
      .emptystate h4 { margin: 0 0 10px 0; font-weight: 700; color:#2f2f2f; }
      .emptystate ul { list-style: none; padding-left: 0; text-align: left; display: inline-block; margin:12px 0 0; }
      .emptystate li::before { content: "💡 "; }

      /* ==== Recommendation grid ==== */
      .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }
    </style>
    """, unsafe_allow_html=True)
  
    # ---------------------------
    # Sidebar
    # ---------------------------
    st.markdown("""
    <style>
    /* 사이드바 전체를 세로 플렉스 + 화면 높이 채우기 */
    aside[data-testid="stSidebar"] > div:first-child {
    height: 100vh;
    display: flex;
    flex-direction: column;
    }

    /* 사이드바 위젯 영역: 기본 패딩 유지(옵션) */
    aside[data-testid="stSidebar"] .sidebar-content {
    padding-right: 0;  /* 필요시 조정 */
    }

    /* 푸터를 맨 아래로 밀기 */
    aside[data-testid="stSidebar"] .sidebar-footer {
    margin-top: auto;
    padding: 12px 8px 16px;
    font-size: 12px;
    color: #6b7280;
    }
    aside[data-testid="stSidebar"] .sidebar-footer hr {
    margin: 8px 0 12px;
    border-color: #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------------ 사이드바 ------------------
    with st.sidebar:
        st.title("옵션")

        # --- 숨김(고정) 설정: UI 없이 강제 적용 ---
        DEFAULT_TODAY_SETTING = os.getenv("TODAY_SETTING", "2024-06-30")
        TODAY_SETTING = DEFAULT_TODAY_SETTING  # 하위 코드 호환용 변수
        st.session_state["n_bm25"] = 80
        st.session_state["top_k"] = 5
        st.session_state["use_hyde"] = True

        # --- 지역 선택 ---
        st.subheader("📍 지역선택")
        cities_available = CITIES
        ALLOWED_CITIES = {"서울"}  # 데모: 서울만

        with st.expander("지역 선택", expanded=True):
            selected_cities = []
            cols = st.columns(2)
            for i, g in enumerate(cities_available):
                is_allowed = g in ALLOWED_CITIES
                default_val = st.session_state.get(f"city_{g}", True if is_allowed else False)
                checked = cols[i % 2].checkbox(
                    g, value=default_val, key=f"city_{g}",
                    disabled=not is_allowed,
                    help=None if is_allowed else "데모에서는 비활성화되어 있어요."
                )
                if checked:
                    selected_cities.append(g)
        st.session_state["selected_cities"] = selected_cities

        # --- 베리어프리 공연장 필터 ---
        st.subheader("♿ 베리어프리")
        bf_only = st.checkbox(
            " 베리어프리 공연장만 보기",
            value=st.session_state.get("bf_only", False),
            help="휠체어석/장애인 화장실 등 접근성 정보를 가진 공연장만 필터링합니다."
        )
        st.session_state["bf_only"] = bf_only

        st.markdown(
            """
            <div class="sidebar-footer">
            <hr/>
            <div>Copyright. 2025 KOPIS 빅데이터 공모전_가보자KU All rights reserved.</div>
            </div>
            """,
            unsafe_allow_html=True
        )



    # ---------------------------
    # Data Loading
    # ---------------------------
    with st.spinner("공연 데이터 불러오는 중..."):
        df_all = load_show_candidates(TODAY_SETTING)
    total_cnt = len(df_all) if df_all is not None else 0

    if "SUBLOC_GAZ" not in st.session_state:
        st.session_state["SUBLOC_GAZ"] = build_address_gazetteer(df_all)
    else:
        st.session_state["SUBLOC_GAZ"] = build_address_gazetteer(df_all)

    # ---------------------------
    # HERO (로고 + 태그라인 + 지표)
    # ---------------------------
    logo_b64 = img_to_base64("assets/CueSense.png")
    logo_html = f'<img class="hero-logo" alt="CueSense" src="data:image/png;base64,{logo_b64}"/>' if logo_b64 else ""

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-left">
            {logo_html}
            <p class="hero-tagline"> 알아서 챙겨주는 공연소식, 센스있게 딱! 큐센스 </p>
          </div>
          <div class="pills">
            <span class="pill">기준일: {pd.to_datetime(TODAY_SETTING).date()}</span>
            <span class="pill">도시: {", ".join(selected_cities) if selected_cities else "선택없음"}</span>
            <span class="pill">공연수: {total_cnt:,}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------
    # Tabs
    # ---------------------------
    st.markdown("##### ")
    tab_reco, tab_help = st.tabs(["✨ 추천", "❓도움말"])

    # 대화 히스토리 준비
    st.session_state.setdefault("chat", [])

    # ============ 탭: 추천 ============
    with tab_reco:

        st.markdown("""
        <style>
        /* 1) 채팅 입력창: 화면 하단 고정 + 메인영역만 차지 */
        div[data-testid="stChatInput"]{
        position: fixed;
        z-index: 999;
        bottom: 20px;                 /* 바닥에서 띄우기 */
        right: 24px;                  /* 우측 여백 */
        left: 360px;                  /* 사이드바 폭(대략)만큼 띄우기: 필요시 조정 */
        max-width: 1100px;            /* 너무 길어지지 않도록 상한 */
        margin: 0 auto;               /* 중앙 정렬 느낌 */
        background: white;
        padding: 0.75rem 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        }

        /* 2) 반응형: 화면이 좁을 땐 좌우 여백만 두고 꽉 차게 */
        @media (max-width: 1100px){
        div[data-testid="stChatInput"]{
            left: 16px;
            right: 16px;
            bottom: 12px;
        }
        }

        /* 3) 내용이 chat_input 뒤에 가리지 않도록 메인 영역에 하단 패딩 추가 */
        section.main, div[data-testid="stAppViewContainer"] section{
        padding-bottom: 140px; /* chat_input 높이 + 여유 */
        }
        </style>
        """, unsafe_allow_html=True)



        # --- 유틸 ---
        def append_chat(role: str, content: str, cards: list | None = None):
            msg = {"role": role, "content": content}
            if cards:
                msg["cards"] = cards
            st.session_state["chat"].append(msg)

        # --- 1) 히스토리 렌더 ---
        for turn in st.session_state["chat"]:
            with st.chat_message("user" if turn["role"] == "user" else "assistant"):
                st.markdown(turn.get("content", ""))

                # ← 카드가 있으면 '직접' 렌더 (그리드 래퍼 없음)
                cards = turn.get("cards") if isinstance(turn, dict) else None
                if cards:
                    for i, it in enumerate(cards, 1):
                        render_card(it, i)


        # --- 2) 입력창  ---
        user_q = st.chat_input("예) 이번 주말 강남 데이트용 바이올린 공연 추천해줘")

        # --- 3) 입력 처리 ---
        if user_q:
            # (A) 유저 메시지를 히스토리에 '먼저' 저장 → 빈 상태 박스 조건 해제
            append_chat("user", user_q)

            # (B) 즉시 화면에도 유저 말풍선 노출(이번 프레임에서 체감)
            with st.chat_message("user"):
                st.markdown(user_q)

            # (C) 로딩 스피너 + 계산
            with st.chat_message("assistant"):
                with st.spinner("딱 맞는 공연을 찾는 중입니다..."):
                    intent = detect_intent(user_q)

                    if intent == "ask_name":
                        # 결과 저장
                        append_chat("assistant", "제 이름은 **CueSence**입니다. 공연 추천을 도와드릴게요!")
                    elif intent != "recommend":
                        append_chat(
                            "assistant",
                            "현재는 **공연 추천/검색** 질문에 최적화되어 있어요.\n\n"
                            "예시) `이번 주말 강남 데이트용 바이올린 공연 추천해줘`, "
                            "`7월 중순 클래식 페스티벌 같은 대형 공연 있어?`"
                        )
                    else:
                        try:
                            try:
                                base_dt = pd.to_datetime(TODAY_SETTING).to_pydatetime()
                            except Exception:
                                base_dt = dt.datetime.now()

                            filters = parse_user_query(
                                user_q, base_dt,
                                subloc_gaz=st.session_state.get("SUBLOC_GAZ", [])
                            )
                            if selected_cities:
                                try:
                                    filters.cities = selected_cities
                                except Exception:
                                    pass

                            df_filt = apply_filters(df_all, filters)
                            df_filt = dedup_by_code(df_filt)

                            if df_filt is None or df_filt.empty:
                                append_chat(
                                    "assistant",
                                    "조건에 맞는 공연이 없어요. 기간/지역/장르를 조금 넓혀볼까요?"
                                )
                            else:
                                ranked, score_map, cand_df = semantic_match(
                                    user_q, df_filt,
                                    n_bm25=st.session_state['n_bm25'],
                                    top_k=st.session_state['top_k'],
                                    use_hyde=st.session_state['use_hyde']
                                )
                                ranked = dedup_ranked_by_code(ranked)
                                ranked = inject_serendipity_slot(
                                    ranked=ranked,
                                    df_filt=df_filt,
                                    score_map=score_map,
                                    k=st.session_state["top_k"],
                                    slot_index=3,
                                    pop_col="popularity",
                                    code_col="performance_code",
                                )

                                if ranked:
                                    append_chat("assistant", "### ✅ 추천 결과", cards=ranked)
                                else:
                                    append_chat(
                                        "assistant",
                                        "필터에 맞는 공연은 있었지만, 시맨틱 조건에 부합하는 결과를 찾지 못했어요."
                                    )
                        except Exception as e:
                            append_chat("assistant", f"오류가 발생했어요: {e}")

            # (D) 히스토리에 저장이 끝났으니, 새 히스토리로 화면 전체 재그리기
            st.rerun()

        # --- 4) 빈 상태 박스 ---
        # chat이 비어 있을 때만 보여줌 (유저 입력 후에는 위에서 append되어 즉시 사라짐)
        if not st.session_state["chat"]:
            st.markdown(
                """
                <div class="emptystate">
                <h4>아직 추천을 시작하지 않았어요</h4>
                <div>아래 예시처럼 물어보세요:</div>
                <ul>
                    <li>이번 주말 서울 데이트용 바이올린 공연 추천해줘</li>
                    <li>7월 중순에 클래식 페스티벌 같은 대형 공연 있어?</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True
            )


    # ============ 탭: 도움말 ============
    with tab_help:
        st.markdown("#### 사용 방법")
        st.markdown(
            """
            - 자연어로 기간·장소·장르·분위기를 함께 적어주세요.  
            - 예) *“이번 주말 강남 데이트용 바이올린 공연 추천해줘”*  
            - 데모에서는 **서울만** 활성화되어 있습니다.
            """
        )

        # 하단 로고 (탭 바닥 고정 느낌)
        help_logo_b64 = img_to_base64("assets/help.png") or img_to_base64("assets/CueSense.png")
        st.markdown(f"""
        <style>
          .help-wrap {{ display:flex; flex-direction:column; min-height: 52vh; }}
          .help-footer {{ margin-top:auto; text-align:center; padding:16px 0 6px; opacity:.9; }}
          .help-footer img {{ height: 40px; filter: drop-shadow(0 1px 4px rgba(0,0,0,.07)); }}
        </style>
        <div class="help-wrap">
          <div class="help-footer">
            {"<img src='data:image/png;base64," + help_logo_b64 + "' alt='CueSense'/>" if help_logo_b64 else ""}
          </div>
        </div>
        """, unsafe_allow_html=True)



if __name__ == "__main__":
    run_app()
