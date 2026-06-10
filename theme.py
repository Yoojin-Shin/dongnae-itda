"""
============================================================================
theme.py — 동네잇다 디자인 시스템 (발표용 v2.0)
============================================================================
지시서 §5 디자인 토큰 + 발표 가독성 규격을 한 곳에서 관리.
사용:
    from theme import COLORS, CLUSTER_COLORS, CONF_COLORS, VAL_COLORS, inject_css
    inject_css()  # st.set_page_config 직후 1회 호출
============================================================================
"""

from __future__ import annotations

import streamlit as st


# ----------------------------------------------------------------------------
# 1. 디자인 토큰
# ----------------------------------------------------------------------------
COLORS = {
    "INK":    "#13294B",  # 헤더·본문 강조
    "TEAL":   "#0E8388",  # Engine 1 · 주 액션
    "AMBER":  "#C77F2A",  # Engine 2 · 가격
    "CORAL":  "#C85A3F",  # 경고 · 고평가
    "GREEN":  "#2E8B6F",  # 저평가 · HIGH
    "PURPLE": "#6B5B95",  # 통합 · DNA
    "MUTED":  "#6B7280",  # 보조 텍스트
    "BG":     "#FAF7F2",
    "CARD":   "#FFFFFF",
    "DIVIDER": "#E5E1D8",
}

CLUSTER_COLORS = {
    0: "#8C99A8",  # C0 도심 평균형
    1: "#21456E",  # C1 고소득 도심 직장
    2: "#6B5B95",  # C2 도심 비거주
    3: "#C77F2A",  # C3 프리미엄 가족
    4: "#2E8B6F",  # C4 주거형 일반
}

# 신뢰도 배지 (FINAL_CONFIDENCE)
CONF_COLORS = {
    "HIGH":          {"bg": "#2E8B6F", "fg": "#FFFFFF", "label": "신뢰 가능",     "icon": "✅"},
    "MEDIUM":        {"bg": "#C77F2A", "fg": "#FFFFFF", "label": "보조 참고용",   "icon": "⚠️"},
    "LOW":           {"bg": "#C85A3F", "fg": "#FFFFFF", "label": "단독 비권장",   "icon": "⚠️"},
    "EXTRAPOLATION": {"bg": "#8B1E1E", "fg": "#FFFFFF", "label": "적용 부적합",   "icon": "❌"},
    "MOCK":          {"bg": "#6B7280", "fg": "#FFFFFF", "label": "Mock 데이터",   "icon": "🧪"},
}

# 평가 배지 (CSV VALUATION: UNDERVALUED / FAIR / OVERVALUED)
VAL_COLORS = {
    "UNDERVALUED": {"bg": "#2E8B6F", "fg": "#FFFFFF", "label": "저평가"},
    "FAIR":        {"bg": "#13294B", "fg": "#FFFFFF", "label": "적정"},
    "OVERVALUED":  {"bg": "#C85A3F", "fg": "#FFFFFF", "label": "고평가"},
}

# 용어 매핑 (지시서 §4 카피 가이드)
GLOSSARY = {
    "match_score": {
        "display": "매칭 점수",
        "tooltip": "취향 방향과 동네 성격의 일치도 (0~1)",
    },
    "price_range": {
        "display": "예상가 범위",
        "tooltip": "실제 가격이 95% 확률로 들어오는 범위",
    },
    "confidence": {
        "display": "신뢰도",
        "tooltip": "범위가 좁을수록 모델이 자신 있는 추정",
    },
    "dna": {
        "display": "동네 DNA",
        "tooltip": "데이터가 발견한 5가지 동네 유형",
    },
}


# ----------------------------------------------------------------------------
# 2. CSS 한 블록 (지시서 §5 — 발표 가독성)
# ----------------------------------------------------------------------------
_CSS = f"""
<style>
:root {{
  --ink: {COLORS['INK']};
  --teal: {COLORS['TEAL']};
  --amber: {COLORS['AMBER']};
  --coral: {COLORS['CORAL']};
  --green: {COLORS['GREEN']};
  --purple: {COLORS['PURPLE']};
  --muted: {COLORS['MUTED']};
  --bg: {COLORS['BG']};
  --card: {COLORS['CARD']};
  --divider: {COLORS['DIVIDER']};
}}

html, body, [class*="css"] {{ font-family: -apple-system, BlinkMacSystemFont, "Pretendard", "Segoe UI", Roboto, sans-serif; }}

/* 발표용 타이포 */
h1, h2 {{ color: var(--ink); letter-spacing: -0.01em; }}
h1 {{ font-size: 32px !important; font-weight: 800; }}
h2 {{ font-size: 24px !important; font-weight: 700; }}
h3 {{ font-size: 20px !important; font-weight: 700; color: var(--ink); }}

/* 동네잇다 카드 (DI-CARD) - 일반용 */
.di-card {{
  background: var(--card);
  border: 1px solid var(--divider);
  border-radius: 14px;
  padding: 20px 22px;
  margin: 0 0 16px 0;
  box-shadow: 0 1px 2px rgba(19,41,75,0.04);
}}

/* ───────────────────────────────────────────────
 * 추천 카드 (DI-REC-CARD) v2 — 반응형 + hover lift
 * ─────────────────────────────────────────────── */
.di-rec-card {{
  background: var(--card);
  border: 1px solid var(--divider);
  border-left: 6px solid var(--purple);
  border-radius: 16px;
  padding: 22px 26px 18px 26px;
  margin: 0 0 14px 0;
  box-shadow: 0 1px 3px rgba(19,41,75,0.06);
  transition: box-shadow .25s ease, transform .25s ease;
  position: relative;
  overflow: hidden;
}}
.di-rec-card:hover {{
  box-shadow: 0 10px 28px rgba(19,41,75,0.10);
  transform: translateY(-2px);
}}
.di-rec-card::before {{
  content: "";
  position: absolute;
  top: 0; right: 0;
  width: 280px; height: 100%;
  background: radial-gradient(circle at top right, rgba(14,131,136,0.05), transparent 70%);
  pointer-events: none;
}}

.di-rec-head {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 10px;
}}
.di-rec-title-wrap {{
  display: flex;
  align-items: baseline;
  gap: 10px;
  flex-wrap: wrap;
}}
.di-rec-rank {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 36px; height: 36px;
  padding: 0 10px;
  background: var(--ink);
  color: #fff;
  font-weight: 800;
  font-size: 16px;
  border-radius: 10px;
  font-variant-numeric: tabular-nums;
}}
.di-rec-name {{
  font-size: 22px;
  font-weight: 800;
  color: var(--ink);
  letter-spacing: -0.01em;
}}
.di-rec-sgg {{
  color: var(--muted);
  font-size: 14px;
  font-weight: 500;
}}

.di-rec-body {{
  display: grid;
  grid-template-columns: minmax(140px, 1.1fr) minmax(280px, 2fr) auto;
  gap: 28px;
  align-items: start;
  margin: 10px 0 6px 0;
}}
@media (max-width: 980px) {{
  .di-rec-body {{
    grid-template-columns: 1fr;
    gap: 16px;
  }}
}}

.di-metric-label {{
  font-size: 11px;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 4px;
}}
.di-metric-value {{
  font-size: 36px;
  font-weight: 800;
  color: var(--ink);
  line-height: 1.05;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}}
.di-metric-unit {{
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
  margin-left: 5px;
  letter-spacing: 0;
}}
.di-metric-sub {{
  font-size: 12px;
  font-weight: 500;
  color: var(--muted);
  margin-top: 4px;
}}

.di-rec-col-badges {{
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 6px;
  min-width: 110px;
}}
@media (max-width: 980px) {{
  .di-rec-col-badges {{ flex-direction: row; align-items: center; }}
}}

/* 핵심 숫자 (일반용 호환) */
.di-bignum {{
  font-size: 30px;
  font-weight: 800;
  color: var(--ink);
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}}
.di-bignum-unit {{
  font-size: 14px;
  font-weight: 600;
  color: var(--muted);
  margin-left: 4px;
}}
.di-subnum {{
  font-size: 16px;
  font-weight: 600;
  color: var(--muted);
}}

/* 배지 v2 — 그림자 + 부드러운 코너 */
.di-badge {{
  display: inline-block;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.02em;
  white-space: nowrap;
  vertical-align: middle;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
.di-badge-dna {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 700;
  color: #fff;
  box-shadow: 0 2px 8px rgba(0,0,0,0.14);
}}

/* 매칭 점수 진행 바 v2 */
.di-progress {{
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(180deg, #ECE7DC, #E5E1D8);
  overflow: hidden;
  margin: 8px 0 2px 0;
  box-shadow: inset 0 1px 2px rgba(19,41,75,0.06);
}}
.di-progress-fill {{
  height: 100%;
  background: linear-gradient(90deg, var(--teal) 0%, var(--purple) 100%);
  border-radius: 999px;
  box-shadow: 0 0 10px rgba(14,131,136,0.35);
  transition: width .7s cubic-bezier(.2,.7,.2,1);
}}

/* 가격 신뢰구간 범위 바 v2 */
.di-rangebar {{
  position: relative;
  width: 100%;
  height: 32px;
  margin: 8px 0 0 0;
}}
.di-rangebar-track {{
  position: absolute;
  left: 0; right: 0;
  top: 14px;
  height: 4px;
  background: linear-gradient(90deg, #ECE7DC, #E5E1D8);
  border-radius: 999px;
}}
.di-rangebar-ci {{
  position: absolute;
  top: 10px;
  height: 12px;
  background: linear-gradient(90deg, rgba(199,127,42,0.65), rgba(200,90,63,0.65));
  border-radius: 999px;
  box-shadow: 0 1px 4px rgba(199,127,42,0.25);
}}
.di-rangebar-point {{
  position: absolute;
  top: 7px;
  width: 18px;
  height: 18px;
  background: var(--ink);
  border-radius: 50%;
  transform: translateX(-9px);
  box-shadow: 0 0 0 3px #fff, 0 0 10px rgba(19,41,75,0.45);
}}
.di-rangebar-labels {{
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  margin-top: 4px;
  font-variant-numeric: tabular-nums;
}}

/* 왜 이 동인가 — 인라인 박스 */
.di-why {{
  background: linear-gradient(180deg, #FAF7F2, #F4EFE3);
  border-radius: 12px;
  padding: 14px 18px;
  margin: 14px 0 0 0;
  border-left: 3px solid var(--teal);
}}
.di-why-h {{
  font-size: 12px;
  font-weight: 800;
  color: var(--ink);
  margin-bottom: 10px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.di-why-item {{
  padding: 10px 0;
  border-bottom: 1px dashed rgba(19,41,75,0.10);
}}
.di-why-item:first-of-type {{ padding-top: 2px; }}
.di-why-item:last-of-type {{ border-bottom: none; padding-bottom: 2px; }}

.di-why-row {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 5px;
  flex-wrap: wrap;
}}
.di-why-emoji {{ font-size: 20px; line-height: 1; }}
.di-why-name {{
  font-weight: 800;
  color: var(--ink);
  font-size: 14.5px;
}}
.di-why-match {{
  margin-left: auto;
  display: inline-block;
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 800;
  color: #fff;
  letter-spacing: 0.04em;
  box-shadow: 0 1px 3px rgba(0,0,0,0.10);
}}
.di-why-detail {{
  font-size: 12.5px;
  color: var(--muted);
  line-height: 1.55;
  padding-left: 30px;
}}
.di-why-detail b {{
  color: var(--ink);
  font-weight: 700;
}}
.di-why-detail .di-tag {{
  display: inline-block;
  padding: 1px 7px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  background: rgba(14,131,136,0.10);
  color: var(--teal);
  margin: 0 2px;
}}
.di-why-detail .di-tag-down {{
  background: rgba(107,91,149,0.10);
  color: var(--purple);
}}

/* 경고 인라인 박스 */
.di-warn {{
  border-radius: 10px;
  padding: 9px 13px;
  font-size: 13px;
  font-weight: 500;
  margin-top: 10px;
}}
.di-warn b {{ font-weight: 800; }}

/* 우선순위 픽 (사이드바) */
.di-prio-pill {{
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  background: var(--teal);
  color: #fff;
  font-size: 12px;
  font-weight: 700;
  margin-right: 6px;
}}
.di-prio-rank {{
  background: var(--ink);
  margin-right: 4px;
}}

/* 데모 프리셋 헤더 */
.di-preset-h {{
  font-size: 13px;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin: 4px 0 6px 0;
}}

/* Streamlit 기본 컴포넌트 오버라이드 */
.stProgress > div > div > div > div {{ background: var(--teal) !important; }}
button[kind="primary"] {{
  background: var(--teal) !important;
  border: none !important;
  font-weight: 700 !important;
}}
button[kind="primary"]:hover {{ background: #0B6A6F !important; }}

/* 사이드바 폭 약간 늘림 */
section[data-testid="stSidebar"] {{ min-width: 340px; }}
</style>
"""


def inject_css() -> None:
    """앱 최상단에서 1회 호출. CSS 토큰·유틸 클래스 일괄 주입."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 3. 인라인 컴포넌트 헬퍼 (HTML 문자열 빌더)
# ----------------------------------------------------------------------------

def progress_bar(pct: float, label: str = "") -> str:
    """매칭 점수 진행 바 HTML."""
    pct = max(0.0, min(100.0, pct))
    return (
        f'<div class="di-progress"><div class="di-progress-fill" '
        f'style="width:{pct:.1f}%;"></div></div>'
    )


def range_bar(
    point: float,
    lower: float,
    upper: float,
    domain_min: float,
    domain_max: float,
) -> str:
    """
    가격 95% CI 범위 바 HTML.

    domain_min~domain_max는 전체 동네 가격 분포 범위 (정규화 기준).
    bar는 [domain_min, domain_max] 도메인에 [lower, upper] CI를 그림.
    """
    if domain_max <= domain_min:
        domain_max = domain_min + 1.0
    span = domain_max - domain_min

    left_pct = max(0.0, (lower - domain_min) / span * 100)
    right_pct = min(100.0, (upper - domain_min) / span * 100)
    width_pct = max(1.5, right_pct - left_pct)
    point_pct = max(0.0, min(100.0, (point - domain_min) / span * 100))

    return (
        '<div class="di-rangebar">'
        '<div class="di-rangebar-track"></div>'
        f'<div class="di-rangebar-ci" style="left:{left_pct:.1f}%; width:{width_pct:.1f}%;"></div>'
        f'<div class="di-rangebar-point" style="left:{point_pct:.1f}%;"></div>'
        '</div>'
        '<div class="di-rangebar-labels">'
        f'<span>{lower:,.0f}</span>'
        f'<span>{upper:,.0f}</span>'
        '</div>'
    )


def conf_badge(confidence: str) -> str:
    """신뢰도 배지 HTML."""
    meta = CONF_COLORS.get(confidence, CONF_COLORS["MEDIUM"])
    return (
        f'<span class="di-badge" '
        f'style="background:{meta["bg"]}; color:{meta["fg"]};" '
        f'title="신뢰도: {meta["label"]}">'
        f'{meta["icon"]} {meta["label"]}'
        f'</span>'
    )


def val_badge(valuation: str, is_imputed: bool = False) -> str:
    """평가 배지 HTML. IMPUTED 동이면 ⚠ 툴팁 부착."""
    meta = VAL_COLORS.get(valuation, VAL_COLORS["FAIR"])
    warn = (
        ' <span title="예측가가 아닌 시군구 평균 대비 판단입니다 (시장가 대비 아님)" '
        'style="cursor:help; color:#C85A3F; font-weight:800;">⚠</span>'
        if is_imputed else ""
    )
    return (
        f'<span class="di-badge" '
        f'style="background:{meta["bg"]}; color:{meta["fg"]};">'
        f'{meta["label"]}'
        f'</span>{warn}'
    )


def dna_badge(cluster_id: int, icon: str, name: str) -> str:
    """DNA 컬러 배지 HTML (카드 우측 상단)."""
    color = CLUSTER_COLORS.get(cluster_id, COLORS["PURPLE"])
    return (
        f'<span class="di-badge-dna" style="background:{color};">'
        f'{icon} {name}'
        f'</span>'
    )
