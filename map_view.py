"""
============================================================================
map_view.py — 동네잇다 지도 시각화 (Plotly choropleth_mapbox + 핀 마커)
============================================================================
118개 법정동을 DNA 클러스터 색상으로 채색한 인터랙티브 지도 + Top N 핀.
mapbox_style="carto-positron"이라 Mapbox 토큰이 필요 없음.

사용:
    from map_view import load_geojson, render_seoul_dna_map
    gj = load_geojson()
    render_seoul_dna_map(data["meta"], gj, pins=result["candidates"])
============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from matching_engine import DNA_LABELS
from theme import CLUSTER_COLORS, COLORS

_GEOJSON_PATH = Path(__file__).parent / "data" / "seoul_dong.geojson"


@st.cache_data
def load_geojson() -> Optional[dict]:
    """data/seoul_dong.geojson 로드 (캐시). 없으면 None."""
    if not _GEOJSON_PATH.exists():
        return None
    with open(_GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def compute_centroids(geojson: dict) -> dict[str, tuple[float, float]]:
    """
    각 동의 polygon centroid (lon, lat) — 핀 마커 위치용.
    Shoelace formula 기반 area-weighted centroid.
    """
    out: dict[str, tuple[float, float]] = {}
    for feat in geojson.get("features", []):
        code = str(feat.get("properties", {}).get("EMD_CD", ""))
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            ring = coords[0]
            cx, cy = _ring_centroid(ring)
        elif gtype == "MultiPolygon":
            # largest polygon's outer ring
            best = None
            best_area = -1.0
            for poly in coords:
                ring = poly[0]
                area = abs(_ring_area(ring))
                if area > best_area:
                    best_area = area
                    best = ring
            cx, cy = _ring_centroid(best) if best else (0.0, 0.0)
        else:
            continue
        out[code] = (cx, cy)
    return out


def _ring_area(ring: list) -> float:
    """Shoelace signed area."""
    n = len(ring)
    s = 0.0
    for i in range(n - 1):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[i + 1][0], ring[i + 1][1]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _ring_centroid(ring: list) -> tuple[float, float]:
    """Area-weighted centroid of a closed ring. Falls back to mean if degenerate."""
    a = _ring_area(ring)
    if abs(a) < 1e-12:
        n = len(ring)
        return (sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n)
    cx = cy = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[i + 1][0], ring[i + 1][1]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    factor = 1.0 / (6.0 * a)
    return (cx * factor, cy * factor)


def render_seoul_dna_map(
    meta_df: pd.DataFrame,
    geojson: dict,
    pins: Optional[List[dict]] = None,
    highlight_codes: Optional[Iterable[str]] = None,
    height: int = 540,
    title: Optional[str] = None,
) -> None:
    """
    서울 118개 동 choropleth — DNA 클러스터 색상 + Top N 핀.

    Args:
        meta_df: dong_metadata.csv (DISTRICT_CODE, DISTRICT_KOR_NAME, SGG, CLUSTER_ID)
        geojson: load_geojson() 반환값
        pins: 추천 결과 candidate dict 리스트.
              각 dict는 rank, district_code, district_kor_name, sgg,
              match_pct, cluster_id 키를 가져야 함. 선택적으로 value_info dict.
        highlight_codes: pins 없이 단순 코드 강조만 할 때 사용 (구버전 호환).
        height: 지도 픽셀 높이
        title: 제목 (None이면 미표시)
    """
    df = meta_df.copy()
    df["DISTRICT_CODE"] = df["DISTRICT_CODE"].astype(str)
    df["cluster_id"] = df["CLUSTER_ID"].astype(int)
    df["dna_name"] = df["cluster_id"].map(lambda c: DNA_LABELS.get(c, {}).get("name", "?"))
    df["dna_icon"] = df["cluster_id"].map(lambda c: DNA_LABELS.get(c, {}).get("icon", ""))

    pin_codes = set()
    if pins:
        pin_codes = {str(p.get("district_code", "")) for p in pins}
    elif highlight_codes:
        pin_codes = {str(c) for c in highlight_codes}

    df["is_pin"] = df["DISTRICT_CODE"].map(lambda c: c in pin_codes)
    df["custom_label"] = df.apply(
        lambda r: (
            f"<b>{r['DISTRICT_KOR_NAME']}</b> · {r['SGG']}<br>"
            f"{r['dna_icon']} C{r['cluster_id']} {r['dna_name']}"
            + ("<br>⭐ <b>추천 동</b>" if r["is_pin"] else "")
        ),
        axis=1,
    )

    fig = go.Figure()

    # 베이스 레이어 — 클러스터별 채색 (강조된 동은 더 진하게)
    for cluster_id in sorted(DNA_LABELS.keys()):
        sub = df[df["cluster_id"] == cluster_id]
        if len(sub) == 0:
            continue
        dna = DNA_LABELS[cluster_id]
        label = f"C{cluster_id} · {dna['icon']} {dna['name']}"
        # 강조 동은 opacity ↑ + 외곽선 굵게
        # 두 레이어로 분리: pinned dongs (강조) + 나머지
        non_pin = sub[~sub["is_pin"]]
        pin = sub[sub["is_pin"]]

        if len(non_pin) > 0:
            fig.add_trace(go.Choroplethmapbox(
                geojson=geojson,
                locations=non_pin["DISTRICT_CODE"],
                z=[cluster_id] * len(non_pin),
                featureidkey="properties.EMD_CD",
                customdata=non_pin[["custom_label"]].values,
                colorscale=[[0, CLUSTER_COLORS[cluster_id]], [1, CLUSTER_COLORS[cluster_id]]],
                showscale=False,
                marker=dict(line=dict(color="#FFFFFF", width=0.7), opacity=0.55),
                hovertemplate="%{customdata[0]}<extra></extra>",
                name=label,
                showlegend=True,
            ))
        if len(pin) > 0:
            fig.add_trace(go.Choroplethmapbox(
                geojson=geojson,
                locations=pin["DISTRICT_CODE"],
                z=[cluster_id] * len(pin),
                featureidkey="properties.EMD_CD",
                customdata=pin[["custom_label"]].values,
                colorscale=[[0, CLUSTER_COLORS[cluster_id]], [1, CLUSTER_COLORS[cluster_id]]],
                showscale=False,
                marker=dict(line=dict(color=COLORS["INK"], width=2.5), opacity=0.92),
                hovertemplate="%{customdata[0]}<extra></extra>",
                name=label,
                showlegend=(len(non_pin) == 0),  # avoid duplicate legend
            ))

    # 핀 마커 레이어 — Top N 추천 동에 순위 번호 표시
    if pins:
        centroids = compute_centroids(geojson)
        lats, lons, texts, hovers, marker_colors = [], [], [], [], []
        for p in pins:
            code = str(p.get("district_code", ""))
            if code not in centroids:
                continue
            lon, lat = centroids[code]
            rank = p.get("rank", "?")
            name = p.get("district_kor_name", "")
            sgg = p.get("sgg", "")
            match_pct = p.get("match_pct", 0)
            cluster_id = int(p.get("cluster_id", 0))
            dna = DNA_LABELS.get(cluster_id, {})
            value = p.get("value_info") or {}
            price_str = (
                f"<br>💰 {value.get('point', 0):,.0f}만원/평"
                if value.get("point") is not None else ""
            )

            lats.append(lat)
            lons.append(lon)
            texts.append(f"<b>{rank}</b>")
            hovers.append(
                f"<b>#{rank} {name}</b> · {sgg}<br>"
                f"매칭 {match_pct:.1f}%<br>"
                f"{dna.get('icon','')} C{cluster_id} {dna.get('name','')}"
                f"{price_str}"
            )
            marker_colors.append(COLORS["INK"])

        if lats:
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons,
                mode="markers+text",
                marker=dict(
                    size=34,
                    color=marker_colors,
                    opacity=0.96,
                ),
                text=texts,
                textfont=dict(size=15, color="#FFFFFF", family="sans-serif"),
                textposition="middle center",
                hovertext=hovers,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                name="추천 핀",
            ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=37.530, lon=126.985),
            zoom=10.4,
        ),
        margin=dict(l=0, r=0, t=(36 if title else 0), b=0),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        title=(dict(text=title, font=dict(size=15)) if title else None),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, Pretendard, sans-serif"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
